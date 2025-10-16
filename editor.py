from pprint import pprint
import math, time, pickle, functools, json, argparse, shutil, os
from enum import Enum, auto
from typing import Union, List, Dict, Tuple
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

argparser = argparse.ArgumentParser(description="Spiritstream")
argparser.add_argument("-f", "--file", type=str, help="Path to file to open")
args = argparser.parse_args()

try:
    with open(WORKSPACEPATH:=Path(__file__).parent / "cache/workspace.json", "r") as f: data=json.load(f)
except: data = {}
if args.file: TEXTPATH, CURSORIDX, CURSORUP, SCENEY = Path(args.file), 0, False, 0
else:
    TEXTPATH, CURSORIDX = Path(data.get("textpath", "text.txt")), data.get("cursoridx", 0)
    CURSORUP, SCENEY = data.get("cursorup", False), data.get("sceney", 0)
CSSPATH = data.get("csspath", "theme.css")
TEXTPATH = TEXTPATH.resolve()

FORMATTING_COLOR = color_from_hex(0xaaa)
CURSOR_COLOR = color_from_hex(0xfff)
SELECTION_COLOR = color_from_hex(0x423024)

HERITABLE_STYLES = ["color", "font-family", "font-size", "line-height", "font-weight", "text-transform"]

class Cursor:
    def __init__(self, frame:Node, idx=0, up=False):
        self.frame = frame # keep reference for cursor_coords and linewraps
        self.idx = idx
        self.pos:vec2
        self.x = None
        self.node:Node = None
        self.selection = Selection()
        
        self.update(idx, up=up)

    def _find(self, node:Node, pos:Union[int, vec2], up:bool=False) -> Node:
        if hasattr(node, "children"):
            if isinstance(pos, int):
                for c in node.children:
                    if c.end is None: show(c)
                    if (up and c.start < pos <= c.end) or (not up and c.start <= pos < c.end): return self._find(c, pos, up)
                return node
            else:
                closest, dx, dy = None, math.inf, math.inf
                # NOTE: cannot recursively descend through hit children because <p><img ...><text>...</text></p> and text wrapping causes wrong hitbox
                for c in walk(node): 
                    if not c.children:
                        if c.x <= pos.x < c.x + c.w and c.y <= pos.y < c.y + c.h: return c # direct hit
                        dxc = 0 if c.x <= pos.x <= c.x+c.w else min(abs(pos.x-c.x), abs(pos.x-(c.x+c.w)))
                        dyc = 0 if c.y <= pos.y <= c.y+c.h else min(abs(pos.y-c.y), abs(pos.y-(c.y+c.h)))
                        if dyc < dy: closest, dx, dy = c, dxc, dyc
                        elif dyc == dy and dxc < dx: closest, dx = c, dxc
                return node if closest is None else closest
        return node

    def update(self, pos:Union[int, vec2], allow_drift:bool=True, up=False, selection=False):
        """
        if pos is vec2, get the int. this might fail because _find returns node.
        In that case, it adds edit=True to the node (and neighbors) and triggers populate_render_data and then updates itself again with pos:vec2
        NOTE: using up=True only works if the line is wrapped. else node.xs[pos-node.start] will be out of range
        """
        if isinstance(pos, int): pos = max(min(pos, len(self.frame.text)), 0)
        if selection and not self.selection: self.selection.i0, self.selection.up0 = self.idx, self.idx == self.node.end
        node = self._find(self.frame.children[0], pos, up=up)
        if node.k is K.LINE:
            if isinstance(pos, int): self.idx, self.pos = pos, vec2(node.xs[pos-node.start], node.y)
            else:
                self.idx = node.start + node.xs.index(x:=min(node.xs, key=lambda x: abs(x-pos.x)))
                self.pos = vec2(x, node.y)
        else:
            node = _find_edit_parent(node)
            rerender = False
            if not _in_edit_view(node): rerender, node.edit = True, Edit.LOCAL
            if rerender:
                populate_render_data(self.frame, SS.css, reset=True)
                self.update(pos if isinstance(pos, int) else node.start, allow_drift=allow_drift, selection=selection)
                return
            else:
                show(node)
                raise RuntimeError(f"Could not resolve position {pos} to LINE after enabling edit mode on {node}")

        self.node = node

        # scroll into view
        # TODO: render only selection that is visible and watch out for edit mode causing jitter while scrolling.
        if self.pos.y + SCENE.y < 0: SCENE.y = -self.pos.y
        elif self.pos.y + SCENE.y + self.node.h > SS.h: SCENE.y = SS.h - self.pos.y - self.node.h

        new_editnodes = set()
        rerender = False
        if selection and self.idx != self.selection.i0:
            self.selection.i1, self.selection.up1 = self.idx, self.idx == self.node.end
            selected_lines = [n for n in walk(self.frame) if n.k is K.LINE
                      and (n.end >= self.selection.start if self.selection.startup else n.end > self.selection.start)
                      and (n.start < self.selection.end if self.selection.endup else n.start <= self.selection.end)]
            for l in selected_lines:
                p = _find_edit_parent(l, l.start)
                if not _in_edit_view(p) and p.parent.k not in [K.EMPTY_LINE]: rerender, p.edit = True, Edit.LOCAL
                if p.edit: new_editnodes.add(p)
        else:
            self.selection.reset()
            SS.quad_buffer_selection.clear()
        p1 = _find_edit_parent(node)
        if p1.parent.k not in [K.EMPTY_LINE]: # edit mode changes nothing here, save some rerenders
            if not _in_edit_view(p1): rerender, p1.edit = True, Edit.LOCAL
            if p1.edit: new_editnodes.add(p1)
        # if cursor between nodes on same line, edit=True on both
        node2 = self._find(self.frame.children[0], self.idx, up=self.idx == node.start)
        p2 = _find_edit_parent(node2)
        if node.y == node2.y and p2.parent.k not in [K.EMPTY_LINE]:
            if not _in_edit_view(p2): rerender, p2.edit = True, Edit.LOCAL
            if p2.edit: new_editnodes.add(p2)

        for n in self.frame.editnodes:
            if n not in new_editnodes: n.edit = None # If frame in editnodes (everthing rendered in edit view), rerender is never True
        if new_editnodes != self.frame.editnodes: rerender = True
        self.frame.editnodes = new_editnodes
        if rerender:
            populate_render_data(self.frame, SS.css, reset=True)
            self.update(self.idx, allow_drift=allow_drift, selection=selection, up=self.idx == self.node.end)
        else:
            if allow_drift: self.x = self.pos.x
            SS.quad_buffer_cursor.replace([*self.pos.components(), 0.4, 2, node.h, (c:=CURSOR_COLOR).r, c.g, c.b, c.a])
            if self.selection:
                selectionQuads = []
                for l in selected_lines:
                    x0 = l.xs[self.selection.start-l.start] if l.start < self.selection.start else l.x
                    x1 = l.xs[self.selection.end-l.start] if l.end > self.selection.end else l.x+l.w
                    selectionQuads.extend([x0-2, l.y, 0.55, x1-x0+4, l.h, (c:=SELECTION_COLOR).r, c.g, c.b, c.a])
                if selectionQuads: SS.quad_buffer_selection.replace(selectionQuads)

    def move(self, d:str, selection):
        assert self.node is not None
        if d == "up":
            pos = vec2(self.pos.x if self.x is None else self.x, self.pos.y)
            leafs = (n for n in walk(self.frame) if not n.children and n.y < self.node.y)
            closest, dx, dy = None, math.inf, math.inf
            for l in leafs:
                dxc = 0 if l.x <= pos.x <= l.x+l.w else min(abs(pos.x-l.x), abs(pos.x-(l.x+l.w)))
                dyc = 0 if l.y <= pos.y <= l.y+l.h else min(abs(pos.y-l.y), abs(pos.y-(l.y+l.h)))
                if dyc < dy: closest, dx, dy = l, dxc, dyc
                elif dyc == dy and dxc < dx: closest, dx = l, dxc
            if closest is None: self.move("start", selection)
            else:
                idx = closest.start + closest.xs.index(min(closest.xs, key=lambda x: abs(x-pos.x))) if closest.k is K.LINE else closest.start
                self.update(idx, selection=selection, allow_drift=False, up=idx==closest.end)
        elif d == "down":
            pos = vec2(self.pos.x if self.x is None else self.x, self.pos.y)
            leafs = (n for n in walk(self.frame) if not n.children and n.y > self.node.y)
            closest, dx, dy = None, math.inf, math.inf
            for l in leafs:
                dxc = 0 if l.x <= pos.x <= l.x+l.w else min(abs(pos.x-l.x), abs(pos.x-(l.x+l.w)))
                dyc = 0 if l.y <= pos.y <= l.y+l.h else min(abs(pos.y-l.y), abs(pos.y-(l.y+l.h)))
                if dyc < dy: closest, dx, dy = l, dxc, dyc
                elif dyc == dy and dxc < dx: closest, dx = l, dxc
            if closest is None: self.move("end", selection)
            else:
                idx = closest.start + closest.xs.index(min(closest.xs, key=lambda x: abs(x-pos.x))) if closest.k is K.LINE else closest.start
                self.update(idx, selection=selection, allow_drift=False, up=idx==closest.end)
        elif d == "start":
            lines = (n for n in walk(self.frame) if n.k is K.LINE and n.y <= self.node.y)
            linestarts, prevy = [], -math.inf
            for l in lines:
                if l.y > prevy:
                    linestarts.append(l)
                    prevy = l.y
            idx, d = None, math.inf
            for l in linestarts:
                if l.y == self.node.y and l.start <= self.idx and (d0:=self.idx-l.start) < d: idx, d = l.start, d0
            if idx is not None: self.update(idx, selection=selection)
            else: raise RuntimeError
        elif d == "end":
            lines = (n for n in walk(self.frame) if n.k is K.LINE and n.y >= self.node.y)
            lineends, end, prevx, prevy = [], None, -math.inf, -math.inf
            for l in lines:
                if l.y == prevy and l.x > prevx: end, prevx, prevy = l, l.x, l.y
                elif l.y > prevy:
                    if end is not None: lineends.append(end)
                    end, prevx, prevy = l, l.x, l.y
            if end is not None: lineends.append(end)
            idx, d = None, math.inf
            for l in lineends:
                if l.y == self.node.y and l.end >= self.idx and (d0:=l.end-self.idx) < d: idx, d = l.end, d0
            if idx is not None:
                # HACK: if text ends with \n, it would try to get to that index, but there is no cursor x position for it.
                if idx in self.frame.wraps or idx == len(self.frame.text) and self.frame.text[-1] != "\n": self.update(idx, selection=selection, up=True)
                else: self.update(idx - 1, selection=selection)
            else: raise RuntimeError

def _in_edit_view(node:Node) -> bool: return _find_edit_parent(node).edit in [Edit.LOCAL, Edit.GLOBAL] or any((n.edit is Edit.GLOBAL for n in parents(node)))
 
def _find_edit_parent(node:Node, idx=None) -> Node: return next((n for n in (node, *parents(node)) if n.k not in INLINE_NODES or n.k is K.CODE and "block" in n.cls))

@dataclass
class Selection:
    i0:int = None
    up0:bool = None # whether this index is correctly found using cursor._find with up=True or not
    i1:int = None
    up1:bool = None
    
    @property
    def start(self): return min(self.i0, self.i1)
    @property
    def end(self): return max(self.i0, self.i1)
    @property
    def startup(self): return self.up0 if self.i0 < self.i1 else self.up1
    @property
    def endup(self): return self.up0 if self.i0 > self.i1 else self.up1

    def reset(self): self.i0 = self.up0 = self.i1 = self.up1 = None
    def copy(self): return Selection(self.i0, self.up0, self.i1, self.up1)
    def __bool__(self): return all((i is not None for i in (self.i0, self.up0, self.i1, self.up1)))

def typeset(text:str, x:float, y:float, width:float, fontfamily:str, fontsize:float, fontweight:int, color:Color, lineheight:float, newline_x:float=None, align="left",
            start=0) -> Tuple[Node, List[float]]:
    lines, idata, wraps = _typeset(text, x, y, width, fontfamily, fontsize, fontweight, color, lineheight, newline_x, align, start)
    SS.glyph_buffer.add(idata)
    frame.wraps.extend(wraps)
    return lines

# TODO: select font more carefully. may contain symbols not in font
# something like g = next((fonŧ[g] for font in fonts if g in font))
# TODO: line_strings is creating ambiguity, sucks, remove. index into text using line start and end to actually get the text. obviously.
@functools.lru_cache(maxsize=1024) # TODO custom cache so it can also shift cached lines down
def _typeset(text:str, x:float, y:float, width:float, fontfamily:str, fontsize:float, fontweight:int, color:Color, lineheight:float, newline_x:float=None, align="left",
             start=0) -> Tuple[Node, List[float]]:
    """
    Returns
    - Node with LINE children to add to the node tree
    - Tex quad instance data for all glyphs
    - List of indices where text wraps
    """
    font, fontweight = get_font(fontfamily, fontweight) # replace fontweight value with actual used based on what's available
    cx = x # character x offset
    linepos = vec2(x, y) # top left corner of line hitbox
    if newline_x is None: newline_x = x
    assert cx <= newline_x + width, (cx, newline_x, width)
    lines, line_strings = [], [""]
    lstart = 0 # char index where line started
    xs = [] # cursor x positions
    wraps = [] # idx of wraps
    idx = 0
    for idx, c in enumerate(text):
        xs.append(cx)
        if c == "\n":
            lines.append(Node(K.LINE, None, x=linepos.x, y=linepos.y, w=cx - linepos.x, h=lineheight, start=start+lstart, end=start + idx+1, xs=xs))
            cx, linepos, xs, lstart = newline_x, vec2(newline_x, linepos.y + lineheight), [], idx+1
            line_strings.append("")
            continue
        g = font.glyph(c, fontsize, 72) # NOTE: dpi is 72 so the font renderer does no further scaling. It uses 72 dpi baseline.
        if cx + g.advance > newline_x + width:
            wrap_idx = fwrap + 1 if (fwrap:=line_strings[-1].rfind(" ")) >= 0 else len(line_strings[-1])
            wraps.append(wrap_idx + start + lstart)
            if wrap_idx == len(line_strings[-1]): # no wrapping necessary, just cut off
                lines.append(Node(K.LINE, None, x=linepos.x, y=linepos.y, w=cx - linepos.x, h=lineheight, start=start+lstart, end=start+idx, xs=xs))
                cx, linepos = newline_x, vec2(newline_x, linepos.y + lineheight)
                xs, lstart = [cx], idx
                line_strings.append("")
            else:
                line_strings.append(line_strings[-1][wrap_idx:])
                line_strings[-2] = line_strings[-2][:wrap_idx]
                lines.append(Node(K.LINE, None, x=linepos.x, y=linepos.y, w=xs[wrap_idx] - linepos.x, h=lineheight, start=start+lstart,
                                end=start+lstart+wrap_idx, xs=xs[:wrap_idx+1]))
                xs, lstart = [newline_x + x0 - xs[wrap_idx] for x0 in xs[wrap_idx:]], lstart + wrap_idx
                cx, linepos = xs[-1], vec2(newline_x, linepos.y + lineheight)

        line_strings[-1] += c
        cx += g.advance

    xs.append(cx) # first position after the last character
    if line_strings[-1] != "" or text == "":
        lines.append(Node(K.LINE, None, x=linepos.x, y=linepos.y, w=cx - linepos.x, h=lineheight, start=start+lstart, end=start + idx+1, xs=xs))
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
    return lines, instance_data, wraps

class Edit(Enum): GLOBAL, LOCAL = auto(), auto() # GLOBAL affects gaps in all descendants, LOCAL affects only gaps in direct children, 

def populate_render_data(node:Node, css_rules:Dict, pstyle:Dict=None, text:str=None, reset:bool=False) -> Tuple[vec2, vec2, vec2]:
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
    
    Returns: Tuple(_block, _inline, _margin) used internally for positioning siblings
    """
    if reset:
        SS.quad_buffer_bg.clear()
        SS.quad_buffer_cursor.clear()
        SS.quad_buffer_selection.clear()
        SS.glyph_buffer.clear()
        for b in SS.img_buffers: b.clear()
        SS.img_buffers.clear()
        to_delete = list(filter(lambda x: x.k is K.LINE, walk(node)))
        for l in to_delete: l.parent.children.remove(l)
        frame.wraps = []
    if node.k is K.FRAME: text = node.text
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
                        style["font-weight"], style["color"], style["line-height"]*1.075, align="right", start=node.start)
            elif node.parent.k is K.OL:
                typeset(d:=f"{node.digit}.", style["_inline"].x-(len(d)+0.5)*style["font-size"], y, style["font-size"]*len(d), style["font-family"],
                        style["font-size"]*1.05, style["font-weight"], style["color"], style["line-height"]*1.075, align="right", start=node.start)
    else:
        style["_width"] = pstyle["_width"]
        style["_margin"] = vec2(style["margin-right"], 0)
        style["_inline"] = pstyle["_inline"] + vec2(style["margin-left"], 0)
        node.x, node.y = style["_inline"].x, style["_inline"].y
        if node.k is K.IMG and not edit_view:
            path = TEXTPATH.parent.joinpath(text[node.href[0]:node.href[1]]).resolve()
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
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img.data)
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
        elif node.k is K.TEXT:
            y = node.y
            if node.parent.k is K.HR and not edit_view: SS.quad_buffer_bg.add([node.x, node.y, 0.6, style["_width"], 1, (c:=style["color"]).r, c.g, c.b, c.a])
            t = text[node.start:node.end].upper() if style.get("text-transform") == "uppercase" else text[node.start:node.end]
            if node.end-node.start > len(t): t += "\n" # HACK: this only happens at file end and ensures that there is an additional cursor position at the end
            lines = typeset(t, node.x, y, style["_width"], style["font-family"], style["font-size"], style["font-weight"], style["color"], style["line-height"],
                            pstyle["_block"].x, start=node.start)
            if lines == []: show(lines); print(t)
            if node.parent.k is K.A:
                font, _ = get_font(style["font-family"], style["font-weight"])
                ascentpx = font.engine.hhea.ascent * style["font-size"] / font.engine.head.unitsPerEM
                descentpx = font.engine.hhea.descent * style["font-size"] / font.engine.head.unitsPerEM
                total = ascentpx - descentpx
                for l in lines:
                    underline_pos = font.engine.fupx(font.engine.post.underlinePosition)
                    underline_y = l.y + ascentpx + (style["line-height"] - total)/2 - underline_pos
                    h = max(1, font.engine.fupx(font.engine.post.underlineThickness))
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
    edit_view_lines = [] # List[Tuple[idx, lines]]
    child_edit_view = edit_view and node.k not in [K.SS, K.SCENE, K.FRAME, K.BODY]
    if node.children:
        for i, child in enumerate(node.children):
            if child.k is not K.LINE:
                if child_edit_view:
                    start = end = None
                    if i == 0 and node.start != child.start: start, end = node.start, child.start
                    elif i > 0 and node.children[i-1].end != child.start: start, end = node.children[i-1].end, child.start
                    if start is not None:
                        # if child.k not in INLINE_NODES: for_children["_inline"].x = pstyle["_block"].x
                        lines = typeset(text[start:end], for_children["_inline"].x, for_children["_inline"].y, style["_width"], style["font-family"],
                                        style["font-size"], style["font-weight"], FORMATTING_COLOR, style["line-height"], for_children["_block"].x, start=start)
                        edit_view_lines.append((i, lines))
                        for_children["_inline"] = vec2((c:=lines[-1]).x + c.w, c.y)
                        for_children["_block"] = vec2(style["_block"].x, c.y + c.h)
                    
                for_children["_block"], for_children["_inline"], for_children["_margin"] = populate_render_data(child, css_rules, for_children, text)

                if i == len(node.children) - 1 and child_edit_view and child.end != node.end:
                    lines = typeset(text[child.end:node.end], for_children["_inline"].x, for_children["_inline"].y, style["_width"], style["font-family"],
                                    style["font-size"], style["font-weight"], FORMATTING_COLOR, style["line-height"], for_children["_block"].x, start=child.end)
                    edit_view_lines.append((i+1, lines))
                    for_children["_inline"] = vec2((c:=lines[-1]).x + c.w, c.y)
                    for_children["_block"] = vec2(style["_block"].x, c.y + c.h)
        for idx, edit_lines in reversed(edit_view_lines): # insertions are delayed until here to preserve insertion idx accuracy
            for l in reversed(edit_lines):
                l.parent = node
                node.children.insert(idx, l)
    elif child_edit_view:
        lines = typeset(text[node.start:node.end], for_children["_inline"].x, for_children["_inline"].y, style["_width"], style["font-family"],
                                    style["font-size"], style["font-weight"], FORMATTING_COLOR, style["line-height"], for_children["_block"].x, start=node.start)
        for l in lines:
            l.parent = node
            node.children.append(l)
        for_children["_inline"] = vec2((c:=node.children[-1]).x + c.w, c.y)
        for_children["_block"] = vec2(style["_block"].x, c.y + c.h)

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

def csspx(pstyle:Dict, style:Dict, k:str) -> float:
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

def queue_events(*events:Tuple):
    for e in events: queue_event(*e)

def queue_event(name, *data):
    """Adds to SS.edit_queue. If name matches latest queue entry, adds data to that data (length must match), else appends a new event"""
    # HACK: mouse_callback and cursor_pos_callback will both cause cursor_pos events. If I just append to the queue, mouse_callback will spam it,
    # if I always replace the latest one, clicking while dragging will replace the cursor_pos_callback with the mouse_callback which always has
    # selection on True, then it would use the previous cursor position for the start of a selection (unintended behavior).
    # allowing max 2 cursor_pos fixes this, allowing click and drag to separate but not spam.
    if name.startswith("cursor_pos") and len(SS.edit_queue) >= 2 and SS.edit_queue[-2][0].startswith("cursor_pos") and\
        SS.edit_queue[-1][0].startswith("cursor_pos"): SS.edit_queue[-1] = [name, *data]
    elif SS.edit_queue and SS.edit_queue[-1][0] == name and name != "open_link": # can't merge open_link
        assert len(SS.edit_queue[-1]) == len(data) + 1
        for i, d in enumerate(data): SS.edit_queue[-1][i+1] += d
    else: SS.edit_queue.append([name, *data])

def pseudoclass(node:Node, pseudocl:str) -> bool: return False # TODO

def selector_match(sel:Selector, node:Node) -> bool: return all((
    sel.k is node.k,
    (all((i in node.cls if node.cls else [] for i in sel.cls))) if sel.cls else True,
    (all((i in node.ids if node.ids else [] for i in sel.ids))) if sel.ids else True,
    (all((pseudoclass(node, i) for i in sel.pseudocls))) if sel.pseudocls else True,
))

def text_update(frame):
    markdown = parse(frame.text)
    frame.children = [markdown]
    markdown.parent = frame
    populate_render_data(frame, SS.css, reset=True)
    glfwSetWindowTitle(window, (TEXTPATH.name + "*").encode())

SS = Node(K.SS, None, resized=True, scrolled=False, fonts={}, glyphAtlas=None, dpi=96, title="Spiritstream", w=700, h=1000, edit_queue=[])
SCENE = Node(K.SCENE, SS, x=0, y=SCENEY)
SS.children = [SCENE]

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
with open(TEXTPATH, "r") as f: markdown = parse(t:=f.read())
with open(CSSPATH, "r") as f: SS.css, SS.fonts = parseCSS(f.read())
default_css = {
    Selector(K.BODY):{"font-size": (16, "px"), "font-weight": 400},
    Selector(K.B):{"font-weight": 700}
}
for rule, declarations in default_css.items():
    for k,v in declarations.items(): SS.css.setdefault(rule, declarations).setdefault(k, v)
frame = Node(K.FRAME, SCENE, [markdown], text=t, editnodes=set(), wraps=[], title=TEXTPATH.stem, edit=False)
SCENE.children = [frame]
markdown.parent = frame

# load fonts TODO: load dynamically when used
for font in SS.fonts:
    assert font["format"] == "truetype"
    font["Font"] = Font(Path(CSSPATH).parent.joinpath(font["url"]).resolve())

populate_render_data(frame, SS.css, reset=True)
frame.cursor = Cursor(frame, idx=CURSORIDX, up=CURSORUP) # after rendertree so it can find line elements in the tree
check(SS)

# show(SS)

while not glfwWindowShouldClose(window):
    # HACK: necessary to actually get multiple events if they accumulated (X11?)
    prevsize = -1

    while len(SS.edit_queue) > prevsize:
        prevsize = len(SS.edit_queue)
        for i in range(20):
            glfwPollEvents()
            time.sleep(0.0001)
    edit_queue = SS.edit_queue.copy()
    SS.edit_queue.clear()

    cursor = frame.cursor.idx
    up = frame.cursor.idx == frame.cursor.node.end
    cursor_selection = frame.cursor.selection.copy()
    clipboard = c.decode() if (c:=glfwGetClipboardString(window)) else ""
    if edit_queue:
        reparse = False
        for name, *data in edit_queue:
            if name == "write":
                if cursor_selection:
                    frame.text = frame.text[:cursor_selection.start] + data[0] + frame.text[cursor_selection.end:]
                    cursor = cursor_selection.start + len(data[0])
                    cursor_selection.reset()
                else:
                    frame.text = frame.text[:cursor] + data[0] + frame.text[cursor:]
                    cursor += len(data[0])
                reparse = True
            elif name == "erase_left":
                if cursor_selection:
                    frame.text = frame.text[:cursor_selection.start] + frame.text[cursor_selection.end:]
                    cursor = cursor_selection.start
                    cursor_selection.reset()
                else:
                    frame.text = frame.text[:max(0, cursor-data[0])] + frame.text[cursor:]
                    cursor = max(0, cursor - data[0])
                reparse = True
            elif name == "erase_right":
                if cursor_selection:
                    frame.text = frame.text[:cursor_selection.start] + frame.text[cursor_selection.end:]
                    cursor = cursor_selection.start
                    cursor_selection.reset()
                else: frame.text = frame.text[:cursor] + frame.text[cursor+data[0]:]
                reparse = True
            # CURSOR
            # must rerender because who knows where the new position is in the changed text
            elif name.startswith("cursor_pos"):
                _, _, selection, x, y = name.split("_")
                selection, x, y = selection == "True", float(x), float(y)
                if reparse: text_update(frame); reparse = False
                frame.cursor.update(vec2(x,y), selection=selection)
                cursor, up, cursor_selection = frame.cursor.idx, frame.cursor.idx == frame.cursor.node.end, frame.cursor.selection.copy()
            elif name == "select_all":
                cursor_selection = Selection(0, False, len(frame.text), False)
                cursor = len(frame.text)
            # START END
            elif name.startswith("start"):
                selection = name.split("_")[1] == "True"
                if reparse: text_update(frame); reparse = False
                frame.cursor.move("start", selection=selection)
                cursor, up, cursor_selection = frame.cursor.idx, frame.cursor.idx == frame.cursor.node.end, frame.cursor.selection.copy()
            elif name.startswith("end"):
                selection = name.split("_")[1] == "True"
                if reparse: text_update(frame); reparse = False
                frame.cursor.move("end", selection=selection)
                cursor, up, cursor_selection = frame.cursor.idx, frame.cursor.idx == frame.cursor.node.end, frame.cursor.selection.copy()
            # CURSOR DIRECTIONS
            elif name.startswith("cursor_up"):
                selection = name.split("_")[2] == "True"
                for i in range(data[0]):
                    if reparse:
                        text_update(frame); reparse = False
                    frame.cursor.move("up", selection=selection)
                cursor, up, cursor_selection = frame.cursor.idx, frame.cursor.idx == frame.cursor.node.end, frame.cursor.selection.copy()
            elif name.startswith("cursor_right"):
                selection = name.split("_")[2] == "True"
                if selection:
                    if cursor_selection: cursor_selection.i1 = min(len(frame.text), cursor+data[0])
                    else: cursor_selection = Selection(cursor, False, min(len(frame.text), cursor+data[0]), False)
                    if (p:=min(len(frame.text), cursor+data[0])) in frame.wraps: up = True
                    cursor = p
                else:
                    if cursor_selection:
                        cursor = cursor_selection.end
                        cursor_selection.reset()
                    else:
                        if (p:=min(len(frame.text), cursor+data[0])) in frame.wraps: up = True
                        cursor = p
            elif name.startswith("cursor_down"):
                selection = name.split("_")[2] == "True"
                for i in range(data[0]):
                    if reparse:
                        text_update(frame); reparse = False
                    frame.cursor.move("down", selection=selection)
                cursor, up, cursor_selection = frame.cursor.idx, frame.cursor.idx == frame.cursor.node.end, frame.cursor.selection.copy()
            elif name.startswith("cursor_left"):
                selection = name.split("_")[2] == "True"
                if selection:
                    if cursor_selection: cursor_selection.i1 = max(0, cursor-data[0])
                    else: cursor_selection = Selection(cursor, False, max(0, cursor-data[0]), False)
                    cursor = max(0, cursor-data[0])
                else:
                    if cursor_selection:
                        cursor = cursor_selection.start
                        cursor_selection.reset()
                    else: cursor = max(0, cursor-data[0])
            # COPY CUT PASTE
            elif name == "copy":
                if cursor_selection: clipboard = frame.text[cursor_selection.start:cursor_selection.end]
            elif name == "cut":
                if cursor_selection:
                    clipboard = frame.text[cursor_selection.start:cursor_selection.end]
                    frame.text = frame.text[:cursor_selection.start] + frame.text[cursor_selection.end:]
                    cursor = cursor_selection.start
                    cursor_selection.reset()
                reparse = True
            elif name == "paste":
                if cursor_selection:
                    frame.text = frame.text[:cursor_selection.start] + clipboard + frame.text[cursor_selection.end:]
                    cursor = cursor_selection.start + len(clipboard)
                    cursor_selection.reset()
                else:
                    frame.text = frame.text[:cursor] + clipboard + frame.text[cursor:]
                    cursor += len(clipboard)
                reparse = True
            # LINK
            elif name == "open_link":
                n = frame.cursor._find(frame.children[0], cursor)
                while n.k in [K.LINE, K.TEXT]: n = n.parent
                if n.k is K.A and n.href:
                    href = frame.text[n.href[0]:n.href[1]]
                    if not href.startswith(("https://", "http://")):
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
                                    with open(p, "r") as f: frame.text = f.read()
                                except:
                                    frame.text = f"Unable to load file {p.as_posix()}"
                                    TEXTPATH = Path("")
                                SCENE.y = 0
                                cursor = 0
                                text_update(frame)
                                glfwSetWindowTitle(window, TEXTPATH.name.encode())
                            if heading:
                                heading = sluggify(heading)
                                for n in walk(frame):
                                    if n.k is K.TEXT and n.parent.k in [K.H1,K.H2,K.H3,K.H4,K.H5,K.H6] and sluggify(frame.text[n.start:n.end]) == heading:
                                        SCENE.y, cursor = -n.y, n.start
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
                            with open(filepath, "r") as f: markdown = parse((t:=f.read()), Node(K.BODY, None, text=t, title=filepath.stem))
                            for n in walk(markdown):
                                if n.k in [K.A, K.IMG] and not (href:=t[n.href[0]:n.href[1]]).startswith(("http://", "https://")):
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
                with open(TEXTPATH.parent.joinpath(f"{TEXTPATH.stem}.html").resolve(), "w") as f: f.write(serialize(frame, Path(CSSPATH), TEXTPATH))
            elif name == "save":
                with open(TEXTPATH, "w") as f: f.write(frame.text)
                glfwSetWindowTitle(window, TEXTPATH.name.encode())
            else: raise NotImplementedError(name, *data)
            
            if name.startswith(("cursor_left", "erase_left")): up = False
            
        if reparse: text_update(frame)
        frame.cursor.selection = cursor_selection.copy()
        frame.cursor.update(cursor, selection=bool(cursor_selection), up=up and cursor in frame.wraps)
        if clipboard: glfwSetClipboardString(window, clipboard.encode()) # copy

    if SS.resized:
        SCENE.x = max(0, (SS.w - frame.w)/2)
        SS.resized = False
        glViewport(0, 0, SS.w, SS.h)
        populate_render_data(frame, SS.css, reset=True)
        frame.cursor.update(frame.cursor.idx, up=up)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    scale, offset = (2 / SS.w, -2 / SS.h), tuple(vec2(-1 + SCENE.x * 2 / SS.w, 1 + SCENE.y * -2 / SS.h).components())

    SS.quad_buffer_bg.draw(scale, offset)
    SS.quad_buffer_cursor.draw(scale, offset)
    SS.quad_buffer_selection.draw(scale, offset)

    SS.glyph_buffer.draw(scale, offset)
    for b in SS.img_buffers: b.draw(scale, offset)
    
    if (error:=glGetError()) != GL_NO_ERROR: print(f"OpenGL Error: {hex(error)}")

    glfwSwapBuffers(window)

with open(WORKSPACEPATH, "w") as f: json.dump({"textpath": TEXTPATH.as_posix(), "sceney": SCENE.y, "cursoridx":frame.cursor.idx,
                                               "cursorup": frame.cursor.idx == frame.cursor.node.end}, f)

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
