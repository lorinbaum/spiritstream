from pprint import pprint
import math, time, pickle, functools, json, argparse, shutil, os, webbrowser
from enum import Enum, auto
from spiritstream.vec import vec2
from spiritstream.bindings.glfw import *
from spiritstream.bindings.opengl import *
from spiritstream.font import Font
from dataclasses import dataclass
from spiritstream.shader import Shader
from spiritstream.textureatlas import TextureAtlas
from spiritstream.helpers import CACHE_GLYPHATLAS, SAVE_GLYPHATLAS
from spiritstream.tree import parse, parseCSS, Node, K, Color, Selector, color_from_hex, walk, show, INLINE_NODES, steal_children, parents, serialize, sluggify, check
from spiritstream.buffer import BaseQuad, QuadBuffer, TexQuadBuffer
import spiritstream.image as Image

# BUG: cutting content puts cursor in wrong positon
# TODO: flickering up and down when selection hits images from below
# TODO: it's exporting fonts even when they not used.

argparser = argparse.ArgumentParser(description="Spiritstream")
argparser.add_argument("file", type=str, nargs="?", default=None, help="Path to file to open")
args = argparser.parse_args()

try:
    with open(WORKSPACEPATH:=Path(__file__).parent / "cache/workspace.json", "r") as f: data=json.load(f)
except: data = {}
if args.file: TEXTPATH, CURSORIDX, SCENEY = Path(args.file), 0, 0
else: TEXTPATH, CURSORIDX, SCENEY = Path(data.get("textpath", "Unnamed.md")), data.get("cursoridx", 0), data.get("sceney", 0)
CURSOR_POST_WRAP = data.get("cursor_post_wrap", False)
CSSPATH = data.get("csspath", "theme.css")
TEXTPATH = TEXTPATH.resolve()

FORMATTING_COLOR = color_from_hex(0xaaa)
CURSOR_COLOR = color_from_hex(0xfff)
SELECTION_COLOR = color_from_hex(0x423024)
SELECTION_Z = 0.55
SELECTION_PADDING = 1

HERITABLE_STYLES = ["color", "font-family", "font-size", "line-height", "font-weight", "text-transform"]

class Cursor:
    # TODO drift. consider write, up in event_queue. no rerender after up so how do I know self.x should have changed?
    def __init__(self, frame:Node, idx=0, post_wrap=False):
        self.frame = frame # keep reference for cursor_coords and linewraps
        self.pos:vec2
        # self.x = None
        self.node:Node = None # K.TEXT
        self.offset = None
        self.selection = None
        self.post_wrap = post_wrap
        self.autogen_len = None

        self._find_index(idx)
        self.update_render_data()

    def _find_index(self, idx):
        """
        Edit can cause additional changes in length of autogen content and so falsify idx. Counteract by keeping track of changes to autogen_len
        before idx and redo with difference added to idx.
        """
        total = 0
        autogen_seen = set()
        for n in walk(self.frame):
            if n.autogen is True:
                for g in filter(lambda x: x.k is K.TEXT and x not in autogen_seen, walk(n)): autogen_seen.add(g)
            if n.k is K.TEXT:
                if total+len(n.text) >= idx and not (idx-total == len(n.text) and n.text[idx-total-1:idx-total] == "\n"):
                    self.autogen_len, old_autogen_len = sum((len(t.text) for t in autogen_seen)), self.autogen_len
                    if self.autogen_len == old_autogen_len or old_autogen_len is None: self.node, self.offset, self.selection = n, idx - total, None
                    else: self._find_index(idx+self.autogen_len-old_autogen_len)
                    return
                total+=len(n.text)
        show(self.frame)
        raise RuntimeError(f"No text node for index {idx} found")
    
    def get_index(self) -> int:
        assert self.node.k is K.TEXT and self.frame in parents(self.node)
        ret = 0
        autogen_seen = set()
        for n in walk(self.frame):
            if n.autogen is True:
                for g in filter(lambda x: x.k is K.TEXT and x not in autogen_seen, walk(n)): autogen_seen.add(g)
            if n.k is K.TEXT:
                if n is self.node: return ret + self.offset, sum((len(t.text) for t in autogen_seen))
                else: ret += len(n.text)
        raise RuntimeError(f"No text in {self.frame=}")

    def _find_pos(self, pos:vec2, selection:bool):
        closest, dx, dy = None, math.inf, math.inf
        for c in walk(self.frame): # TODO can be more efficient if descending block node hitboxes before leafnodes only
            if c.k is K.LINE or (c.k is K.IMG and not any((n.k is K.LINE for n in walk(c)))):
                if c.x <= pos.x < c.x + c.w and c.y <= pos.y < c.y + c.h:
                    closest = c # direct hit
                    break
                dxc = 0 if c.x <= pos.x <= c.x+c.w else min(abs(pos.x-c.x), abs(pos.x-(c.x+c.w)))
                dyc = 0 if c.y <= pos.y <= c.y+c.h else min(abs(pos.y-c.y), abs(pos.y-(c.y+c.h)))
                if dyc < dy: closest, dx, dy = c, dxc, dyc
                elif dyc == dy and dxc < dx: closest, dx = c, dxc
        self._update_from_closest(closest, pos.x, selection)

    def _update_from_closest(self, closest:Node, x:int, selection:bool):
        assert closest.k in [K.LINE, K.IMG], closest
        if closest.k is K.LINE:
            node, line_offset = closest.parent, closest.xs.index(min(closest.xs, key=lambda x1: abs(x1-x)))
            offset = sum((len(n.xs)-1 for n in node.children[:node.children.index(closest)])) + line_offset
            post_wrap = line_offset == 0
        else: node, offset, post_wrap = closest.children[0], len(closest.children[0].text), False
        self.update_location(node, offset, selection, post_wrap)

    def next(self, node=None): return next((n for n in walk(self.node if node is None else node, siblings=True) if n.k is K.TEXT), None)
    def prev(self, node=None): return next((n for n in walk(self.node if node is None else node, reverse=True, siblings=True) if n.k is K.TEXT), None)

    def update_render_data(self):
        """
        if self.post_wrap is True and a given offset is shared by two lines, updating render data will use the second line, else the first (default).
        """
        assert self.node.k is K.TEXT
        # update edit view
        new_editnodes = set()
        new_editnodes.add(_find_edit_parent(self.node))
        if self.offset == 0 and (prev:=self.prev()) is not None and _find_edit_parent(self.node) is _find_edit_parent(prev): new_editnodes.add(_find_edit_parent(prev))
        if self.offset == len(self.node.text) and (nxt:=self.next()) is not None and _find_edit_parent(self.node) is _find_edit_parent(nxt): new_editnodes.add(_find_edit_parent(nxt))
        new_editnodes = set(filter(lambda node: not any((n.edit is Edit.GLOBAL for n in (node, *parents(node)))), new_editnodes))

        if self.selection:
            selected = set()
            if self.selection.start.node is not self.selection.end.node:
                gen = (n for n in walk(self.selection.start.node, siblings=True) if n.k is K.TEXT)
                while (nxt:=next(gen, None)):
                    if nxt is self.selection.end.node: break
                    selected.add(nxt)
            for n in (self.selection.start.node, *selected, self.selection.end.node): new_editnodes.add(_find_edit_parent(n))

        if self.frame.editnodes != new_editnodes:
            for n in self.frame.editnodes: n.edit = False
            self.frame.editnodes = new_editnodes
            for n in new_editnodes: n.edit = Edit.LOCAL
            populate_render_data(self.frame, SS.css)
            self.update_render_data()
        else:
            l, x = self.get_line_x()
            # scroll into view
            if l.y + SCENE.y < 0: SCENE.y = -l.y
            elif l.y + SCENE.y + l.h > SS.h: SCENE.y = SS.h - l.y - l.h
            # TODO: render only selection that is visible and watch out for edit mode causing flashing while scrolling.
            if self.selection:
                c,p,z = SELECTION_COLOR, SELECTION_PADDING, SELECTION_Z
                selectionQuads = []
                start_l, start_x = self.get_line_x(self.selection.start.node, self.selection.start.offset, self.selection.start.post_wrap)
                end_l, end_x = self.get_line_x(self.selection.end.node, self.selection.end.offset, self.selection.end.post_wrap)
                if start_l is not end_l:
                    # add selection start
                    selectionQuads.extend([start_x-p, start_l.y, z, start_l.xs[-1]-start_x+2*p, start_l.h, c.r, c.g, c.b, c.a])
                    # add inbetween
                    gen = (l for l in walk(start_l, siblings=True) if l.k is K.LINE)
                    while (nxt:=next(gen, None)):
                        if nxt is end_l: break
                        selectionQuads.extend([nxt.x-p, nxt.y, z, nxt.w+2*p, nxt.h, c.r, c.g, c.b, c.a])
                    # add selection end
                    selectionQuads.extend([end_l.x-p, end_l.y, z, end_x-end_l.x+2*p, end_l.h, c.r, c.g, c.b, c.a])
                else: selectionQuads.extend([start_x-p, start_l.y, z, end_x-start_x+2*p, start_l.h, c.r, c.g, c.b, c.a]) # add selection start
                if selectionQuads: SS.quad_buffer_selection.replace(selectionQuads)
            else: SS.quad_buffer_selection.clear()
            SS.quad_buffer_cursor.replace([x, l.y, 0.4, 2, l.h, (c:=CURSOR_COLOR).r, c.g, c.b, c.a])
    
    def get_line_x(self, node=None, offset=None, post_wrap=None) -> tuple[Node, float]:
        # NOTE: if text content was modified, this is only safe to call if the text was also reparsed, else offset can be out of range of line xs.
        node, offset, post_wrap = map(lambda i: getattr(self, i[0]) if i[1] is None else i[1], (("node", node), ("offset", offset), ("post_wrap", post_wrap)))
        total, x = 0, None
        if not (node.k is K.TEXT and node.children): raise RuntimeError()
        for l in node.children:
            if (post_wrap and (total+len(l.xs)-1) > offset) or (not post_wrap and (total+len(l.xs)-1) >= offset):
                x = l.xs[offset-total]
                break
            total += len(l.xs)-1
        if x is None and post_wrap and total == offset: x = l.xs[-1] # if post_wrap prevents finding x but no post_wrap would be valid, use last x
        if l is None or x is None:
            show(self.frame)
            raise RuntimeError((node, l, x, total, offset, post_wrap))
        return l, x

    def update_location(self, node:Node, offset:int, selection:bool, post_wrap:bool=False):
        """Sets the parameters as attributes of self while considering any previous or new selections."""
        if selection:
            if self.selection: self.selection.l1 = Location(node, offset, post_wrap)
            else: self.selection = Selection(Location(self.node, self.offset, self.post_wrap), Location(node, offset, post_wrap))
        elif self.selection: self.selection = None
        self.node, self.offset, self.post_wrap = node, offset, post_wrap
    
    def right(self, count:int=1, selection:bool=False):
        # NOTE: if not parsed from markdown, can become unintuitive if line starts with \n\n
        for _ in range(count):
            if self.selection and not selection: node, offset = self.selection.end.node, self.selection.end.offset
            else:
                if (nxt:=self.next()) is None: node, offset = self.node, min(self.offset+1, len(self.node.text))
                elif self.offset == len(self.node.text): node, offset = nxt, 1
                else: node, offset = self.node, self.offset+1
                # TODO: this should probably be in update_location to prohibit placing a cursor right of \n in any case
                if nxt is not None and node.text[offset-1:offset] == "\n" and offset==len(node.text):
                    if nxt is node and (nxt2:=self.next(nxt)) is not None: nxt = nxt2
                    node, offset = nxt, 0
            self.update_location(node, offset, selection, self.selection.end.post_wrap if self.selection else node.text[offset-1:offset] == "\n")

    def left(self, count:int=1, selection:bool=False):
        for _ in range(count):
            if self.selection and not selection: node, offset = self.selection.start.node, self.selection.start.offset
            elif (prev:=self.prev()) is None: node, offset = self.node, max(self.offset-1, 0)
            elif self.offset == 0: node, offset = prev, len(prev.text)-1
            else: node, offset = self.node, self.offset-1
            self.update_location(node, offset, selection, self.selection.start.post_wrap if self.selection else True)

    def up(self, count:int=1, selection:bool=False):
        if self.selection and not selection:
            self.node, self.offset, self.post_wrap = self.selection.start.node, self.selection.start.offset, self.selection.start.post_wrap
        prev, x = self.get_line_x()
        for _ in range(count):
            node = next((n for n in walk(prev, reverse=True, siblings=True) if n.k in [K.LINE, K.IMG] and n.y+n.h<=prev.y), None)
            if node is None:
                self.update_location(prev.parent if prev.k is K.LINE else prev.children[0], 0, selection)
                self.start(selection)
                return
            else: prev = node
        upline = {prev} # include original line
        while (prev:=next((n for n in walk(prev, reverse=True, siblings=True) if n.k in [K.LINE, K.IMG] and not n.y+n.h<=prev.y), None)): upline.add(prev)
        closest = min((n for n in upline), key=lambda n: min([abs(n.x-x), abs(n.x+n.w-x), 0 if n.x<=x<n.x+n.w else math.inf]))
        self._update_from_closest(closest, x, selection)

    def down(self, count:int=1, selection:bool=False):
        if self.selection and not selection:
            self.node, self.offset, self.post_wrap = self.selection.end.node, self.selection.end.offset, self.selection.end.post_wrap
        prev, x = self.get_line_x()
        for _ in range(count):
            node = next((n for n in walk(prev, siblings=True) if n.k in [K.LINE, K.IMG] and n.y>=prev.y+prev.h), None)
            if node is None:
                self.update_location((node:=prev.parent if prev.k is K.LINE else prev.children[0]), len(node.text), selection)
                self.end(selection)
                return
            else: prev = node
        downline = {prev} # include original line
        while (prev:=next((n for n in walk(prev, siblings=True) if n.k in [K.LINE, K.IMG] and not n.y>=prev.y+prev.h), None)): downline.add(prev)
        closest = min((n for n in downline), key=lambda n: min([abs(n.x-x), abs(n.x+n.w-x), 0 if n.x<=x<n.x+n.w else math.inf]))
        self._update_from_closest(closest, x, selection)

    def start(self, selection:bool):
        (prev, _), start = self.get_line_x(), None
        while prev is not None:
            start, g = prev, (n for n in walk(prev, reverse=True, siblings=True) if n.k in [K.LINE, K.IMG])
            prev = None if (n:=next(g, None)) is None or n.y+n.h <= prev.y else n
        assert start is not None
        self._update_from_closest(start, start.x, selection)

    def end(self, selection:bool):
        (prev, _), end = self.get_line_x(), None
        while prev is not None:
            end, g = prev, (n for n in walk(prev, siblings=True) if n.k in [K.LINE, K.IMG])
            prev = None if (n:=next(g, None)) is None or n.y >= prev.y+prev.h else n
        assert end is not None
        self._update_from_closest(end, end.x+end.w, selection)

def _in_edit_view(node:Node) -> bool: return _find_edit_parent(node).edit in [Edit.LOCAL, Edit.GLOBAL] or any((n.edit is Edit.GLOBAL for n in parents(node)))
 
def _find_edit_parent(node:Node) -> Node:
    if (nxt:=next((n for n in (node, *parents(node)) if n.k is K.TOC), None)): return nxt
    return next((n for n in (node, *parents(node)) if n.k not in INLINE_NODES or n.k is K.CODE and "block" in n.cls))

@dataclass(frozen=True)
class Location:
    node:Node
    offset:int
    post_wrap:bool

class Selection:
    def __init__(self, l0:Location, l1:Location):
        self.l0, self.l1 = l0, l1
    def __setattr__(self, name, value:Location):
        assert name in ["l0", "l1"] and isinstance(value, Location), (name, value) and value.node.k is K.TEXT
        super().__setattr__(name, value)
        if hasattr(self, "l0") and hasattr(self, "l1"):
            start, end = self._order(self.l0, self.l1)
            for k,v in (("start", start), ("end", end)): super().__setattr__(k,v)
    def _order(self, l0:Location, l1:Location) -> tuple:
        """Returns in order in which locations l0 and l1 appear in the tree. Uses offset if nodes are the same, else Lowest Common Ancestor"""
        if l0.node is l1.node: return (l0, l1) if l0.offset < l1.offset else (l1, l0)
        p0, p1 = map(lambda n: list((n, *parents(n))), (l0.node, l1.node))
        if p0[0] in p1 or p1[0] in p0: raise RuntimeError(f"Can't determine order, {l0=} or {l1=} is inside the other.")
        p0, p1 = p0[-(l:=min(len(p0), len(p1))):], p1[-l:]
        idx, lca = next(((i, i0) for i,(i0,i1) in enumerate(zip(p0, p1)) if i0 is i1))
        n = next((n for n in lca.children if n in (p0[idx-1], p1[idx-1])))
        return (l0, l1) if n is p0[idx-1] else (l1, l0)
    def __repr__(self): return f"Selection(\n  {self.start}\n  {self.end})"

def _get_selection_text(frame:Node, sel:Selection) -> str:
    assert frame in parents(sel.start.node) and frame in parents(sel.end.node)
    if sel.start.node is sel.end.node: return sel.start.node.text[sel.start.offset:sel.end.offset]
    else:
        ret = sel.start.node.text[sel.start.offset:]
        gen = (n for n in walk(sel.start.node, siblings=True) if n.k is K.TEXT)    
        while (nxt:=next(gen, None)):
            if nxt is sel.end.node: break
            ret += nxt.text
        return ret + sel.end.node.text[:sel.end.offset]
    
def _del_selection_text(frame:Node, sel:Selection):
    assert frame in parents(sel.start.node) and frame in parents(sel.end.node)
    if sel.start.node is sel.end.node: sel.start.node.text = sel.start.node.text[:sel.start.offset] + sel.start.node.text[sel.end.offset:]
    else:
        sel.start.node.text = sel.start.node.text[:sel.start.offset]
        gen = (n for n in walk(sel.start.node, siblings=True) if n.k is K.TEXT)    
        while (nxt:=next(gen, None)) and nxt is not sel.end.node: nxt.text = ""
        sel.end.node.text = sel.end.node.text[sel.end.offset:]

def typeset(text:str, x:float, y:float, width:float, fontfamily:str, fontsize:float, fontweight:int, color:Color, lineheight:float, newline_x:float=None, align="left") -> tuple[Node, list[float]]:
    lines, idata = _typeset(text, x, y, width, fontfamily, fontsize, fontweight, color, lineheight, newline_x, align)
    SS.glyph_buffer.add(idata)
    return lines

# TODO: select font more carefully. may contain symbols not in font
# something like g = next((fonŧ[g] for font in fonts if g in font))
# TODO: line_strings is creating ambiguity, sucks, remove. index into text using line start and end to actually get the text. obviously.
@functools.lru_cache(maxsize=1024) # TODO custom cache so it can also shift cached lines down
def _typeset(text:str, x:float, y:float, width:float, fontfamily:str, fontsize:float, fontweight:int, color:Color, lineheight:float, newline_x:float=None, align="left") -> tuple[Node, list[float]]:
    """
    Returns
    - Node with LINE children to add to the node tree
    - Tex quad instance data for all glyphs
    - list of indices where text wraps
    """
    font, fontweight = get_font(fontfamily, fontweight) # replace fontweight value with actual used based on what's available
    cx = x # character x offset
    linepos = vec2(x, y) # top left corner of line hitbox
    if newline_x is None: newline_x = x
    assert cx <= newline_x + width, (cx, newline_x, width)
    lines, line_strings = [], [""]
    xs = [cx] # cursor x positions
    for i, c in enumerate(text):
        if c == "\n":
            lines.append(Node(K.LINE, None, x=linepos.x, y=linepos.y, w=cx - linepos.x, h=lineheight, xs=xs+[cx]))
            cx, linepos, xs = newline_x, vec2(newline_x, linepos.y + lineheight), [] if i == len(text)-1 else [newline_x]
            line_strings.append("")
            continue
        g = font.glyph(c, fontsize, 72) # NOTE: dpi is 72 so the font renderer does no further scaling. It uses 72 dpi baseline.
        if cx + g.advance > newline_x + width:
            wrap_idx = fwrap + 1 if (fwrap:=line_strings[-1].rfind(" ")) >= 0 else len(line_strings[-1])
            if wrap_idx == len(line_strings[-1]): # no wrapping necessary, just cut off
                lines.append(Node(K.LINE, None, x=linepos.x, y=linepos.y, w=cx - linepos.x, h=lineheight, xs=xs))
                cx, linepos = newline_x, vec2(newline_x, linepos.y + lineheight)
                xs = [cx]
                line_strings.append("")
            else:
                line_strings.append(line_strings[-1][wrap_idx:])
                line_strings[-2] = line_strings[-2][:wrap_idx]
                lines.append(Node(K.LINE, None, x=linepos.x, y=linepos.y, w=xs[wrap_idx] - linepos.x, h=lineheight, xs=xs[:wrap_idx+1]))
                xs = [newline_x + x0 - xs[wrap_idx] for x0 in xs[wrap_idx:]]
                cx, linepos = xs[-1], vec2(newline_x, linepos.y + lineheight)

        line_strings[-1] += c
        cx += g.advance
        xs.append(cx)

    if xs: lines.append(Node(K.LINE, None, x=linepos.x, y=linepos.y, w=cx - linepos.x, h=lineheight, xs=xs))
    if align in ["center", "right"]:
        for c in lines:
            shift = x+width-c.w-c.x / (2 if align == "center" else 1)
            c.x, c.xs = c.x + shift, [x0 + shift for x0 in c.xs]

    instance_data = []
    ascentpx = font.engine.hhea.ascent * fontsize / font.engine.head.unitsPerEM
    descentpx = font.engine.hhea.descent * fontsize / font.engine.head.unitsPerEM
    total = ascentpx - descentpx
    for s, l in zip(line_strings, lines):
        for i, c in enumerate(s):
            g = font.glyph(c, fontsize, 72)
            if c != " ":
                # TODO: font could be named the same in css but refer to a different file. use font name from file here, not css font family
                key = f"{ord(c)}_{fontfamily}_{fontsize}_{fontweight}"
                instance_data += [l.xs[i] + g.bearing.x, l.y + ascentpx + (lineheight - total)/2 + (g.size.y - g.bearing.y), 0.5, # pos
                    g.size.x, -g.size.y, # size
                    *(SS.glyphAtlas.coordinates[key] if key in SS.glyphAtlas.coordinates else SS.glyphAtlas.add(key, font.render(c, fontsize, 72))), # uv offset and size
                    color.r, color.g, color.b, color.a] # color
    return lines, instance_data

class Edit(Enum): GLOBAL, LOCAL = auto(), auto() # GLOBAL affects gaps in all descendants, LOCAL affects only gaps in direct children, 

def populate_render_data(node:Node, css_rules:dict, pstyle:dict=None):
    SS.quad_buffer_bg.clear()
    SS.quad_buffer_cursor.clear()
    SS.quad_buffer_selection.clear()
    SS.glyph_buffer.clear()
    for b in SS.img_buffers: b.clear()
    SS.img_buffers.clear()
    to_delete = list(filter(lambda x: x.k is K.LINE, walk(node)))
    for l in to_delete: l.parent.children.remove(l)
    for n in walk(node):
        if n.k is K.TEXT: n.x, n.y, n.w, n.h = None, None, None, None # formatting text that is no longer visible could disrupt cursor finding position.
    _populate_render_data(node, css_rules, pstyle)

def _populate_render_data(node:Node, css_rules:dict, pstyle:dict=None) -> tuple[vec2, vec2, vec2]:
    """
    Recursively walks through node and children to:
    - determine style information for each node
    - call typeset to:
        - populate quad_instance_data and tex_quad_instance_data
        - add Line nodes to the tree for interactive text
        - set x, y, w, h properties of each node
    
    Parameters:
        - pstyle (parent style) holds information specific to the parent's children, like inherited styles and below's internal properties

    4 internal properties help with positioning:
        _block:vec2     absolute position of next child / sibling if block element: for child if passed in pstyle argument, for siblings if returned
        _inline:vec2    same for inline elements
        _margin:vec2    Used for margin collapse. _margin.x = margin-right of latest inline element, _margin.y = margin-bottom of latest block element
        _width:float    width of content box, passed to children
    
    Returns: tuple(_block, _inline, _margin) used internally for positioning siblings
    """
    # inherit and apply style
    if pstyle is None: pstyle = {"font-size": (16.0, "px")}
    for k, v in {"_block": vec2(0, 0), "_inline": vec2(0, 0), "_margin": vec2(0, 0), "_width": 0.0}.items(): pstyle.setdefault(k, v)
    assert all([isinstance(v, vec2) for v in [pstyle["_block"], pstyle["_inline"], pstyle["_margin"]]]) and isinstance(pstyle["_width"], float)
    style = {k:v for k, v in pstyle.items() if k in HERITABLE_STYLES}
    style.update({k:v for sel, rules in css_rules.items() for k, v in rules.items() if sel.k is K.ANY})
    style.update({k:v for sel, rules in css_rules.items() for k, v in rules.items() if selector_match(sel, node)}) # TODO: sel like p code
    style.update({k:csspx(pstyle, style, k) for k in ["width", "font-size"] if k in style}) # update first, other relative units need them
    for_children = {k:v for k,v in style.items() if k in HERITABLE_STYLES} # pass on before more conversion so relative values are recomputed
    style = {k:csspx(pstyle, style, k) for k in style.keys()} # convert px, em, %, auto to device pixels
    
    edit_view = _in_edit_view(node)

    style["display"] = style.get("display", "inline" if node.k in INLINE_NODES else "block")
    assert style["display"] in ["inline", "block"]
    # TODO: check syntax in parsing, not processing. maybe should be function to verify generally, use in testing
    # process block / inline elements
    if style["display"] == "block":
        assert pstyle.get("display", "block") == "block"
        node.w = style.get("width", pstyle["_width"]) - style["margin-left"] - style["margin-right"]
        style["_width"] = node.w - style["padding-left"] - style["padding-right"]
        assert style["_width"] >= 0
        node.x, node.y = pstyle["_block"].x + style["margin-left"], pstyle["_block"].y + max(style["margin-top"], pstyle["_margin"].y)
        style.update({"_block": (v:=vec2(node.x + style["padding-left"], node.y + style["padding-top"])), "_inline": v, "_margin": vec2(0,0)})
        if node.k is K.LI and not edit_view:
            y = node.y + style["font-size"] * 0.066666 # hack to approximate browser rendering. Shift without changing node.y for talled node.
            if node.parent.k is K.UL:
                typeset("•", style["_inline"].x - 1.25 * style["font-size"], y, style["font-size"], style["font-family"], style["font-size"]*1.4,
                        style["font-weight"], style["color"], style["line-height"]*1.075, align="right")
            elif node.parent.k is K.OL:
                typeset(d:=f"{node.digit}.", style["_inline"].x-(len(d)+0.5)*style["font-size"], y, style["font-size"]*len(d), style["font-family"],
                        style["font-size"]*1.05, style["font-weight"], style["color"], style["line-height"]*1.075, align="right")
    else:
        style["_width"] = pstyle["_width"]
        style["_margin"] = vec2(style["margin-right"], 0)
        style["_inline"] = pstyle["_inline"] + vec2(style["margin-left"], 0)
        node.x, node.y = style["_inline"].x, style["_inline"].y
        if node.k is K.IMG and not edit_view:
            path = TEXTPATH.parent.joinpath(node.href).resolve()
            if not path.exists(): path = Path(__file__).parent / "home/Image-not-found.png"
            if path.as_posix() not in SS.img_cache:
                img = Image.read(path)
                assert img.color
                texture = ctypes.c_uint()
                glGenTextures(1, ctypes.byref(texture))
                glBindTexture(GL_TEXTURE_2D, texture)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                fmt = GL_RGBA if img.alpha else GL_RGB
                glTexImage2D(GL_TEXTURE_2D, 0, fmt, img.width, img.height, 0, fmt, GL_UNSIGNED_BYTE, img.data)
                glBindTexture(GL_TEXTURE_2D, 0)
                img = SS.img_cache[path.as_posix()] = {"width": img.width, "height": img.height, "buffer": TexQuadBuffer(SS.base_quad, texture, img_shader)}
            else: img = SS.img_cache[path.as_posix()]
            node.w = min(style["_width"], img["width"])
            if node.x - pstyle["_block"].x + node.w > style["_width"]: node.x, node.y = pstyle["_block"].x, node.y + style["line-height"]
            node.h = node.w / img["width"] * img["height"]
            img["buffer"].add([node.x, node.y, 0.5, node.w, node.h])
            SS.img_buffers.add(img["buffer"])

            font, _ = get_font(style["font-family"], style["font-weight"])
            ascentpx = font.engine.hhea.ascent * style["font-size"] / font.engine.head.unitsPerEM
            descentpx = font.engine.hhea.descent * style["font-size"] / font.engine.head.unitsPerEM
            total = ascentpx - descentpx
            style["_inline"] = vec2(node.x + node.w, node.y + node.h - (style["line-height"] - total)/2 - ascentpx)
            style["_block"] = vec2(pstyle["_block"].x, style["_inline"].y + style["line-height"])
            return style["_block"], style["_inline"], vec2(0, style["margin-bottom"]) # early return because img never has children
        elif node.k is K.TEXT and ((edit_view and "formatting" in node.cls) or "formatting" not in node.cls):
            y = node.y
            if node.parent.k is K.HR and not edit_view: SS.quad_buffer_bg.add([node.x, node.y, 0.6, style["_width"], 1, (c:=style["color"]).r, c.g, c.b, c.a])
            t = node.text.upper() if style.get("text-transform") == "uppercase" else node.text
            color = FORMATTING_COLOR if edit_view and "formatting" in node.cls else style["color"]
            lines = typeset(t, node.x, y, style["_width"], style["font-family"], style["font-size"], style["font-weight"], color, style["line-height"],
                            pstyle["_block"].x)
            if node.parent.k is K.A:
                font, _ = get_font(style["font-family"], style["font-weight"])
                ascentpx = font.engine.hhea.ascent * style["font-size"] / font.engine.head.unitsPerEM
                descentpx = font.engine.hhea.descent * style["font-size"] / font.engine.head.unitsPerEM
                total = ascentpx - descentpx
                for l in lines:
                    underline_pos = font.engine.fupx(font.engine.post.underlinePosition, style["font-size"], 72)
                    underline_y = l.y + ascentpx + (style["line-height"] - total)/2 - underline_pos
                    h = max(1, font.engine.fupx(font.engine.post.underlineThickness, style["font-size"], 72))
                    SS.quad_buffer_bg.add([l.x, underline_y, 0.6, l.w, h, (c:=style["color"]).r, c.g, c.b, c.a])
            for l in lines:
                l.parent = node
                node.children.append(l)
            c = node.children[-1]
            style["_block"] = vec2(pstyle["_block"].x, c.y + c.h)
            style["_inline"] = style["_block"].copy() if t.endswith("\n") else vec2(c.x + c.w, c.y)
        else: style["_block"] = pstyle["_block"]

    # process children and if edit view, fill any gaps (omitted formatting text) before and after
    # NOTE: Formatting text is put OUTSIDE of TEXT nodes because their start/end is outside. Expanding TEXT start/end would falsify content
    for_children.update({k:v for k,v in style.items() if k in ["_block", "_inline", "_margin", "_width"]})
    if node.children:
        for child in node.children:
            if child.k is not K.LINE:
                for_children["_block"], for_children["_inline"], for_children["_margin"] = _populate_render_data(child, css_rules, for_children)
    style.update({k:v for k,v in for_children.items() if k in ["_block", "_inline", "_margin", "_width"]})

    # update _block, _inline and _margin for return
    if style["display"] == "block":
        if node.children:
            if not getattr(node, "w", None): node.w = max([c.x + c.w for c in node.children]) - node.x # applies to frame node
            node.h = (end := style["_block"].y + style["padding-bottom"]) - node.y
            # node.h = (end := node.children[-1].y + node.children[-1].h + style["padding-bottom"]) - node.y
            if (c:=style.get("background-color")) and isinstance(c, Color): # ignoring "black", "red" and such
                SS.quad_buffer_bg.add([node.x, node.y, 0.6, node.w, node.h, (c:=style["background-color"]).r, c.g, c.b, c.a])
            return (_block:=vec2(pstyle["_block"].x, end)), _block, vec2(0, style["margin-bottom"]) # _block, _inline, _margin
        else:
            node.h = style["padding-top"] + style["padding-bottom"]
            return (_block:=pstyle["_block"] + vec2(0, style["padding-bottom"]), _block, vec2(0, style["margin-bottom"]))
    else:
        # backgroundcolor not supported
        if node.children:
            node.w = max([c.x + c.w for c in node.children]) - node.x
            node.h = style["_block"].y - node.y
        else:
            node.w = node.h = 0
            style["_block"] = pstyle["_block"]
        return style["_block"], style["_inline"], style["_margin"]
    
def get_font(fontfamily:str, fontweight:int) -> Font:
    ret, d, w = None, math.inf, None
    for font in SS.fonts:
        if font["font-family"] == fontfamily and (d0:=abs((w0:=font.get("font-weight", 400))-fontweight)) < d: ret, d, w = font["Font"], d0, w0
    assert ret is not None
    return ret, w

def csspx(pstyle:dict, style:dict, k:str) -> float:
    """Converts CSS values based on unit. CSS assumes 96 DPI for px and EM are relative to font-size"""
    v = style[k]
    if v == "auto" and k in ["margin-left", "margin-right"]: return max((pstyle["_width"] - style["width"]) / 2, 0) # style["width"] must be defined
    if v == "auto" and k in ["margin-top", "margin-bottom"]: return 0 # style["width"] must be defined
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], (int, float)) and isinstance(v[1], str):
        v, u = v
        if u == "px": return v * SS.dpi / 96
        # if font-size or _width not defined yet, these must be the properties currently computed and they should use pstyle values if they are relative
        elif u == "em": return (pstyle["font-size"] if k == "font-size" else style["font-size"]) * v
        elif u == "%": return (pstyle["font-size"] if k == "font-size" else pstyle["_width"]) * v / 100
    return v[0] if isinstance(v, tuple) and len(v) == 1 else v

@GLFWframebuffersizefun
def framebuffer_size_callback(window, width, height): SS.w, SS.h, SS.resized = width, height, True

@GLFWcharfun
def char_callback(window, codepoint): queue_event("write", chr(codepoint))

@GLFWkeyfun
def key_callback(window, key:int, scancode:int, action:int, mods:int):
    if action in [GLFW_PRESS, GLFW_REPEAT]:
        shift = bool(mods & GLFW_MOD_SHIFT)
        ctrl = bool(mods & GLFW_MOD_CONTROL)
        if key == GLFW_KEY_LEFT: queue_event(f"cursor_left_{shift}", 1)
        elif key == GLFW_KEY_RIGHT: queue_event(f"cursor_right_{shift}", 1)
        elif key == GLFW_KEY_UP: queue_event(f"cursor_up_{shift}", 1)
        elif key == GLFW_KEY_DOWN: queue_event(f"cursor_down_{shift}", 1)
        
        elif key == GLFW_KEY_BACKSPACE: queue_event("erase_left", 1)
        elif key == GLFW_KEY_DELETE: queue_event("erase_right", 1)
        elif key == GLFW_KEY_ENTER and ctrl: queue_event("open_link")
        elif key == GLFW_KEY_ENTER: queue_event("write", "\n")

        elif key == GLFW_KEY_C and mods & GLFW_MOD_CONTROL and frame.cursor.selection: queue_event("copy")
        elif key == GLFW_KEY_V and mods & GLFW_MOD_CONTROL: queue_event("write", glfwGetClipboardString(window).decode()) # paste
        elif key == GLFW_KEY_X and mods & GLFW_MOD_CONTROL and frame.cursor.selection: queue_event("cut") # cut

        elif key == GLFW_KEY_HOME: queue_event(f"start_{shift}")
        elif key == GLFW_KEY_END: queue_event(f"end_{shift}")

        elif key == GLFW_KEY_S and ctrl: queue_event("save")
        elif key == GLFW_KEY_A and mods & GLFW_MOD_CONTROL: queue_event("select_all")
        elif key == GLFW_KEY_ESCAPE: queue_event("close")
        elif key == GLFW_KEY_E and ctrl and shift: queue_events(("save",), ("full_export",))
        elif key == GLFW_KEY_E and ctrl: queue_event("export")

@GLFWmousebuttonfun
def mouse_callback(window, button:int, action:int, mods:int):
    if button == GLFW_MOUSE_BUTTON_1 and action is GLFW_PRESS:
        x, y = ctypes.c_double(), ctypes.c_double()
        glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
        pos, selection = vec2(x.value - SCENE.x, y.value - SCENE.y), glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS
        queue_event(f"cursor_pos_{selection}_{pos.x}_{pos.y}")
        if mods & GLFW_MOD_CONTROL: queue_event("open_link")

@GLFWcursorposfun
def cursor_pos_callback(window, x, y):
    if glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS:
        x, y = ctypes.c_double(), ctypes.c_double()
        glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
        queue_event(f"cursor_pos_True_{x.value - SCENE.x}_{y.value - SCENE.y}")

@GLFWscrollfun
def scroll_callback(window, x, y): SCENE.y += y * 40

def queue_events(*events:tuple):
    for e in events: queue_event(*e)

def queue_event(name, *data):
    """Adds to SS.event_queue. If name matches latest queue entry, adds data to that data (length must match), else appends a new event"""
    # HACK: mouse_callback and cursor_pos_callback will both cause cursor_pos events. If I just append to the queue, mouse_callback will spam it,
    # if I always replace the latest one, clicking while dragging will replace the cursor_pos_callback with the mouse_callback which always has
    # selection on True, then it would use the previous cursor position for the start of a selection (unintended behavior).
    # allowing max 2 cursor_pos fixes this, allowing click and drag to separate but not spam.
    if name.startswith("cursor_pos") and len(SS.event_queue) >= 2 and SS.event_queue[-2][0].startswith("cursor_pos") and\
        SS.event_queue[-1][0].startswith("cursor_pos"): SS.event_queue[-1] = [name, *data]
    elif SS.event_queue and SS.event_queue[-1][0] == name and name != "open_link": # can't merge open_link
        assert len(SS.event_queue[-1]) == len(data) + 1
        for i, d in enumerate(data): SS.event_queue[-1][i+1] += d
    else: SS.event_queue.append([name, *data])

def pseudoclass(node:Node, pseudocl:str) -> bool: return False # TODO

def selector_match(sel:Selector, node:Node) -> bool: return all((
    sel.k is node.k,
    (all((i in node.cls if node.cls else [] for i in sel.cls))) if sel.cls else True,
    (all((i in node.ids if node.ids else [] for i in sel.ids))) if sel.ids else True,
    (all((pseudoclass(node, i) for i in sel.pseudocls))) if sel.pseudocls else True,
))

def text_update(frame, text:str=None):
    if text is None: (idx, autogen_len), markdown = frame.cursor.get_index(), parse("".join((n.text for n in walk(frame, autogen=False) if n.k is K.TEXT)))
    else: idx, autogen_len, markdown = 0, 0, parse(text)
    frame.children = [markdown]
    markdown.parent = frame
    populate_render_data(frame, SS.css)
    frame.cursor.autogen_len = autogen_len
    frame.cursor._find_index(idx)
    glfwSetWindowTitle(window, (TEXTPATH.name + "*").encode())

SS = Node(K.SS, None, resized=True, scrolled=False, fonts={}, glyphAtlas=None, dpi=96, title="Spiritstream", w=700, h=1000, event_queue=[])
SCENE = Node(K.SCENE, SS, x=0, y=SCENEY)
SS.children.append(SCENE)

glfwInit()
glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, True)
window = glfwCreateWindow(SS.w, SS.h, SS.title.encode("utf-8"), None, None)
glfwMakeContextCurrent(window)

if CACHE_GLYPHATLAS:
    atlaspath = Path(__file__).parent / "cache/TextureAtlas.pkl"
    atlaspath.parent.mkdir(parents=True, exist_ok=True)
    if atlaspath.exists():
        with open(atlaspath, "rb") as f: SS.glyphAtlas = pickle.load(f)
        SS.glyphAtlas._texture_setup()
    else: print(f"No cached glyphatlas found at {atlaspath}.")
# GL_RED because single channel, assign after window creation + assignment else texture settings won't apply
if not getattr(SS, "glyphAtlas", None): SS.glyphAtlas = TextureAtlas(GL_RED)

glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)
glfwSetCharCallback(window, char_callback)
glfwSetKeyCallback(window, key_callback)
glfwSetMouseButtonCallback(window, mouse_callback)
glfwSetCursorPosCallback(window, cursor_pos_callback)
glfwSetScrollCallback(window, scroll_callback)

glyph_shader = Shader(p:=Path(__file__).parent / "spiritstream/shaders/glyph.vert", p.parent / "glyph.frag", ["glyphAtlas", "scale", "offset"])
glyph_shader.setUniform("glyphAtlas", 0, "1i")  # 0 means GL_TEXTURE0)
quad_shader = Shader(p:=Path(__file__).parent / "spiritstream/shaders/quad.vert", p.parent / "quad.frag", ["scale", "offset"])
img_shader = Shader(p:=Path(__file__).parent / "spiritstream/shaders/img.vert", p.parent / "img.frag", ["scale", "offset"])

SS.base_quad = BaseQuad()
SS.quad_buffer_bg = QuadBuffer(SS.base_quad, quad_shader)
SS.quad_buffer_cursor = QuadBuffer(SS.base_quad, quad_shader)
SS.quad_buffer_selection = QuadBuffer(SS.base_quad, quad_shader)
SS.glyph_buffer = TexQuadBuffer(SS.base_quad, SS.glyphAtlas.texture, glyph_shader, tinted_atlas=True)
SS.img_buffers = set() # Set[TexQuadBuffer] only images to be rendered
SS.img_cache = {}

glActiveTexture(GL_TEXTURE0)
glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

glEnable(GL_DEPTH_TEST)
glClearDepth(1)
glClearColor(0, 0, 0, 1)

TEXTPATH.touch()
glfwSetWindowTitle(window, TEXTPATH.name.encode())
with open(TEXTPATH, "r") as f: markdown = parse(f.read())
with open(CSSPATH, "r") as f: SS.css, SS.fonts = parseCSS(f.read())
default_css = {
    Selector(K.BODY):{"font-size": (16, "px"), "font-weight": 400},
    Selector(K.B):{"font-weight": 700}
}
for rule, declarations in default_css.items():
    for k,v in declarations.items(): SS.css.setdefault(rule, declarations).setdefault(k, v)
frame = Node(K.FRAME, SCENE, [markdown], editnodes=set(), wraps=[], title=TEXTPATH.stem, edit=False)
SCENE.children = [frame]
markdown.parent = frame

# load fonts TODO: load dynamically when used
for font in SS.fonts:
    assert font["format"] == "truetype"
    font["Font"] = Font(Path(CSSPATH).parent.joinpath(font["url"]).resolve())

populate_render_data(frame, SS.css)
# show(SS)
frame.cursor = Cursor(frame, idx=CURSORIDX, post_wrap=CURSOR_POST_WRAP) # after rendertree so it can find line elements in the tree
check(SS)

# show(SS)

while not glfwWindowShouldClose(window):
    # HACK: necessary to actually get multiple events if they accumulated (X11?)
    prevsize = -1
    while len(SS.event_queue) > prevsize:
        prevsize = len(SS.event_queue)
        for i in range(20):
            glfwPollEvents()
            time.sleep(0.0001)
    event_queue = SS.event_queue.copy()
    SS.event_queue.clear()

    if event_queue:
        reparse = False
        for name, *data in event_queue:
            if name == "write":
                if frame.cursor.selection:
                    _del_selection_text(frame, frame.cursor.selection)
                    frame.cursor.update_location((s:=frame.cursor.selection.start).node, s.offset, False, s.post_wrap)
                cn, o = frame.cursor.node, frame.cursor.offset
                cn.text = cn.text[:o] + data[0] + cn.text[o:]
                frame.cursor.right(len(data[0]))
                reparse = True
            elif name == "erase_left":
                if frame.cursor.selection:
                    _del_selection_text(frame, frame.cursor.selection)
                    frame.cursor.update_location((s:=frame.cursor.selection.start).node, s.offset, False, s.post_wrap)
                else:
                    for _ in range(data[0]):
                        if frame.cursor.offset == 0:
                            if (prev:=frame.cursor.prev()):
                                prev.text = prev.text[:-1]
                                frame.cursor.update_location(prev, len(prev.text), False)
                            else: break
                        else:
                            cn = frame.cursor.node
                            cn.text = cn.text[:max(0, frame.cursor.offset-1)] + cn.text[frame.cursor.offset:]
                            frame.cursor.offset = frame.cursor.offset-1
                            frame.cursor.post_wrap = True # always True when walking left
                reparse = True
            elif name == "erase_right":
                for _ in range(data[0]):
                    if frame.cursor.selection:
                        _del_selection_text(frame, frame.cursor.selection)
                        frame.cursor.update_location((s:=frame.cursor.selection.start).node, s.offset, False, s.post_wrap)
                    else:
                        if frame.cursor.offset == len(frame.cursor.node.text):
                            if (nxt:=frame.cursor.next()): nxt.text = nxt.text[1:]
                            else: break
                        else:
                            cn = frame.cursor.node
                            cn.text = cn.text[:frame.cursor.offset] + cn.text[frame.cursor.offset+1:]
                reparse = True
            # CURSOR
            # must rerender because who knows where the new position is in the changed text
            elif name.startswith("cursor_pos"):
                _, _, selection, x, y = name.split("_")
                selection, x, y = selection == "True", float(x), float(y)
                if reparse: text_update(frame); reparse = False
                frame.cursor._find_pos(vec2(x,y), selection)
            elif name == "select_all": frame.cursor.selection = Selection(Location(next((n for n in walk(frame) if n.k is K.TEXT)), 0, False),
                Location((n:=next((n for n in walk(frame, reverse=True) if n.k is K.TEXT))), len(n.text), True))
            # START END
            elif name.startswith("start"):
                if reparse: text_update(frame); reparse = False
                frame.cursor.start(name.split("_")[1] == "True")
            elif name.startswith("end"):
                if reparse: text_update(frame); reparse = False
                frame.cursor.end(name.split("_")[1] == "True")
            # CURSOR DIRECTIONS
            elif name.startswith("cursor_up"):
                if reparse: text_update(frame); reparse = False
                frame.cursor.up(data[0], name.split("_")[2] == "True")
            elif name.startswith("cursor_right"): frame.cursor.right(data[0], name.split("_")[2] == "True")
            elif name.startswith("cursor_down"):
                if reparse: text_update(frame); reparse = False
                frame.cursor.down(data[0], name.split("_")[2] == "True")
            elif name.startswith("cursor_left"): frame.cursor.left(data[0], name.split("_")[2] == "True")
            # COPY CUT PASTE
            elif name == "copy":
                if frame.cursor.selection: glfwSetClipboardString(window, _get_selection_text(frame, frame.cursor.selection).encode())
            elif name == "cut":
                if frame.cursor.selection:
                    glfwSetClipboardString(window, _get_selection_text(frame, frame.cursor.selection).encode())
                    _del_selection_text(frame, frame.cursor.selection)
                    frame.cursor.update_location((s:=frame.cursor.selection.start).node, s.offset, False, s.post_wrap)
                    reparse = True
            elif name == "paste":
                if frame.cursor.selection:
                    _del_selection_text(frame, frame.cursor.selection)
                    frame.cursor.update_location((s:=frame.cursor.selection.start).node, s.offset, False, s.post_wrap)
                cn, o, t = frame.cursor.node, frame.cursor.offset, glfwGetClipboardString(window)
                cn.text = cn.text[:o] + t + cn.text[o:]
                frame.cursor.right(len(t))
                reparse = True
            # LINK
            elif name == "open_link":
                n = frame.cursor.node
                while n.k in [K.LINE, K.TEXT]: n = n.parent
                if n.k is K.A:
                    href = n.href
                    if href.startswith(("https://", "http://")): webbrowser.open(href)
                    else:
                        hashidx = href.rfind("#")
                        if hashidx == -1: path, heading = href, ""
                        else: path, heading = href[:hashidx], href[hashidx+1:]
                        try: href = TEXTPATH.parent.joinpath(path).resolve() if path != "" else TEXTPATH.resolve()
                        except: continue
                        if not (p:=Path(href)).is_absolute():
                            try: p = TEXTPATH.parent.joinpath(p).resolve()
                            except: continue
                        if p.exists():
                            if p.as_posix() != TEXTPATH.as_posix():
                                TEXTPATH = p
                                try:
                                    with open(p, "r") as f: t = f.read()
                                except:
                                    t = f"Unable to load file {p.as_posix()}"
                                    TEXTPATH = Path("")
                                SCENE.y = 0
                                text_update(frame, t)
                                glfwSetWindowTitle(window, TEXTPATH.name.encode())
                            if heading:
                                heading = sluggify(heading)
                                for n in walk(frame):
                                    if n.k in [K.H1,K.H2,K.H3,K.H4,K.H5,K.H6] and sluggify("".join((c.text for c in walk(n) if c.k is K.TEXT and "formatting" not in c.cls)))==heading:
                                        SCENE.y = -n.y
                                        frame.cursor.update_location(next((c for c in walk(n) if c.k is K.TEXT and "formatting" not in c.cls)), 0, False)
                                        break
                        else: print(f"INVALID LINK: Target does not exist: {p}")
            # META
            elif name == "close": glfwSetWindowShouldClose(window, GLFW_TRUE)
            elif name == "full_export":                
                to_export = {TEXTPATH.resolve()}
                seen = set() # paths to source of exported files
                targets = set() # paths to output of exported files
                
                out = Path(__file__).parent / "output/"
                if out.exists(): shutil.rmtree(out)
                os.mkdir(out)
                
                csspath = Path(shutil.copy(CSSPATH, out / "style.css"))
                targets.add(csspath)

                while len(diff:=to_export.difference(seen)) > 0:
                    for filepath in diff:
                        if filepath.suffix == ".md": # convert to html and get additional files this points to
                            with open(filepath, "r") as f: markdown = parse((t:=f.read()), Node(K.BODY, None, title=filepath.stem))
                            for n in walk(markdown):
                                if n.k in [K.A, K.IMG] and not (href:=n.href).startswith(("http://", "https://")):
                                    if n.k is K.A:
                                        path, heading = (href, "") if (hashidx:=href.rfind("#")) == -1 else (href[:hashidx], href[hashidx+1:])
                                        try: href = filepath.parent.joinpath(path).resolve() if path != "" else filepath.resolve()
                                        except: continue
                                    else:
                                        try: href = filepath.parent.joinpath(href).resolve() if not (href:=Path(href)).is_absolute() else href
                                        except: continue
                                    if href.exists(): to_export.add(href)
                            with open((target:=out / (filepath.stem + ".html")), "w") as f: f.write(serialize(markdown, csspath, target))
                        else: shutil.copy(filepath, (target:=out / filepath.name))
                        targets.add(target)
                        seen.add(filepath)

                for font in SS.fonts: # move fonts
                    fontpath = csspath.parent.joinpath(Path(font["url"])).resolve() # assumes url is not from the web and is relative
                    assert out in iter(fontpath.parents)
                    os.makedirs(fontpath.parent, exist_ok=True)
                    shutil.copy(Path(CSSPATH).parent.joinpath(Path(font["url"])).resolve(), fontpath)
                    targets.add(fontpath)

                print("Fully exported:", *sorted(targets, key=lambda x: x.suffix), sep="\n    ")

            elif name == "export":
                with open((out:=TEXTPATH.parent.joinpath(f"{TEXTPATH.stem}.html").resolve()), "w") as f: f.write(serialize(frame, Path(CSSPATH), TEXTPATH))
                print(f"Exported to {out}")
            elif name == "save":
                t = "".join((n.text for n in walk(frame, autogen=False) if n.k is K.TEXT))
                with open(TEXTPATH, "w") as f: f.write(t)
                glfwSetWindowTitle(window, TEXTPATH.name.encode())
                SCENEY = SCENE.y
                CURSORIDX, _ = frame.cursor.get_index()
                CURSOR_POST_WRAP = frame.cursor.post_wrap
            else: raise NotImplementedError(name, *data)
            
            if name.startswith(("cursor_left", "erase_left")): up = False
            
        if reparse: text_update(frame)
        frame.cursor.update_render_data()

    if SS.resized:
        SCENE.x = max(0, (SS.w - frame.w)/2)
        SS.resized = False
        glViewport(0, 0, SS.w, SS.h)
        populate_render_data(frame, SS.css)
        frame.cursor.update_render_data()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    scale, offset = (2 / SS.w, -2 / SS.h), tuple(vec2(-1 + SCENE.x * 2 / SS.w, 1 + SCENE.y * -2 / SS.h).components())

    SS.quad_buffer_bg.draw(scale, offset)
    SS.quad_buffer_selection.draw(scale, offset)
    SS.glyph_buffer.draw(scale, offset)
    SS.quad_buffer_cursor.draw(scale, offset)

    for b in SS.img_buffers: b.draw(scale, offset)
    
    if (error:=glGetError()) != GL_NO_ERROR: print(f"OpenGL Error: {hex(error)}")

    glfwSwapBuffers(window)

with open(WORKSPACEPATH, "w") as f: json.dump({"textpath": TEXTPATH.as_posix(), "sceney": SCENEY, "cursoridx": CURSORIDX, "cursor_post_wrap": CURSOR_POST_WRAP}, f)

if SAVE_GLYPHATLAS: Image.write(list(reversed(SS.glyphAtlas.bitmap)), Path(__file__).parent / "GlyphAtlas.bmp")

if CACHE_GLYPHATLAS:
    with open(atlaspath, "wb") as f: pickle.dump(SS.glyphAtlas, f)

SS.base_quad.delete()
SS.quad_buffer_bg.delete()
SS.quad_buffer_cursor.delete()
SS.quad_buffer_selection.delete()
SS.glyph_buffer.delete()
for i in SS.img_cache.values(): i["buffer"].delete()
glyph_shader.delete()
quad_shader.delete()
glfwTerminate()
