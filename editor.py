from pprint import pprint
import math, time, pickle, functools, json, argparse
from typing import Union, List, Dict, Tuple
from spiritstream.vec import vec2
from spiritstream.bindings.glfw import *
from spiritstream.bindings.opengl import *
from spiritstream.font import Font
from dataclasses import dataclass
from spiritstream.shader import Shader
from spiritstream.textureatlas import TextureAtlas
from spiritstream.helpers import CACHE_GLYPHATLAS, SAVE_GLYPHATLAS
from spiritstream.tree import parse, parseCSS, Node, K, Color, Selector, color_from_hex, walk, show, INLINE_NODES, steal_children

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

FORMATTING_COLOR = color_from_hex(0xaaa)
CURSOR_COLOR = color_from_hex(0xfff)
SELECTION_COLOR = color_from_hex(0x423024)

class Cursor:
    def __init__(self, frame:Node, idx=0, up=False):
        self.frame = frame # keep reference for cursor_coords and linewraps
        self.idx = idx
        self.pos:vec2
        self.x = None
        self.node:Node
        self.selection = Selection()
        
        self.update(idx, up=up)

    def _find(self, node:Node, pos:Union[int, vec2], up:bool=False) -> Node:
        if hasattr(node, "children"):
            if isinstance(pos, int):
                for c in node.children:
                    if c.start <= pos and ((pos <= c.end and up) or (not up and pos < c.end)): return self._find(c, pos, up)
                return node if up  or not node.children else self._find(node, pos, True) # try up before giving up
            else:
                closest, dx, dy = None, math.inf, math.inf
                for c in node.children:
                    if c.x <= pos.x < c.x + c.w and c.y <= pos.y < c.y + c.h: return self._find(c, pos) # direct hit
                    dxc = 0 if c.x <= pos.x <= c.x+c.w else min(abs(pos.x-c.x), abs(pos.x-(c.x+c.w)))
                    dyc = 0 if c.y <= pos.y <= c.y+c.h else min(abs(pos.y-c.y), abs(pos.y-(c.y+c.h)))
                    if dyc < dy: closest, dx, dy = c, dxc, dyc
                    elif dyc == dy and dxc < dx: closest, dx = c, dxc
                return self._find(closest, pos) if closest is not None else node
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
            if isinstance(pos, int):
                self.idx, self.pos = pos, vec2(node.xs[pos-node.start], node.y)
            else:
                self.idx = node.start + node.xs.index(x:=min(node.xs, key=lambda x: abs(x-pos.x)))
                self.pos = vec2(x, node.y)
        else:
            assert isinstance(pos, int)
            node = _find_edit_parent(node, pos)
            rerender = False
            if not _in_edit_view(node): rerender = node.edit = True
            if rerender:
                populate_render_data(self.frame, SS.css, reset=True)
                self.update(pos, allow_drift=allow_drift, selection=selection)
            else: raise RuntimeError(f"Could not resolve position {pos} to LINE after enabling edit mode on {node}")
            return

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
                if not _in_edit_view(p) and p.parent.k not in [K.EMPTY_LINE]: rerender = p.edit = True
                if p.edit: new_editnodes.add(p)
        else:
            self.selection.reset()
            QuadBuffer.replace([], "selection")
        p1 = _find_edit_parent(node, self.idx)
        if p1.parent.k not in [K.EMPTY_LINE]: # edit mode changes nothing here, save some rerenders
            if not _in_edit_view(p1): rerender = p1.edit = True
            if p1.edit: new_editnodes.add(p1)
        # if cursor between nodes on same line, edit=True on both
        node2 = self._find(self.frame.children[0], self.idx, up=self.idx == node.start)
        p2 = _find_edit_parent(node2, self.idx)
        if node.y == node2.y and p2.parent.k not in [K.EMPTY_LINE]:
            if not _in_edit_view(p2): rerender = p2.edit = True
            if p2.edit: new_editnodes.add(p2)

        for n in self.frame.editnodes:
            if n not in new_editnodes: n.edit = False # If frame in editnodes (everthing rendered in edit view), rerender is never True
        if new_editnodes != self.frame.editnodes: rerender = True
        self.frame.editnodes = new_editnodes
        if rerender:
            populate_render_data(self.frame, SS.css, reset=True)
            self.update(self.idx, allow_drift=allow_drift, selection=selection)
        else:
            if allow_drift: self.x = self.pos.x
            QuadBuffer.replace([*self.pos.components(), 0.4, 2, node.h, (c:=CURSOR_COLOR).r, c.g, c.b, c.a], "cursor")
            if self.selection:
                selectionQuads = []
                for l in selected_lines:
                    x0 = l.xs[self.selection.start-l.start] if l.start < self.selection.start else l.x
                    x1 = l.xs[self.selection.end-l.start] if l.end > self.selection.end else l.x+l.w
                    selectionQuads.extend([x0-2, l.y, 0.55, x1-x0+4, l.h, (c:=SELECTION_COLOR).r, c.g, c.b, c.a])
                if selectionQuads: QuadBuffer.replace(selectionQuads, "selection")

    def move(self, d:str, selection):
        assert self.node is not None
        if d == "up":
            pos = vec2(self.pos.x if self.x is None else self.x, self.pos.y)
            lines = (n for n in walk(self.frame) if n.k is K.LINE and n.y < self.node.y)
            closest, dx, dy = None, math.inf, math.inf
            for l in lines:
                dxc = 0 if l.x <= pos.x <= l.x+l.w else min(abs(pos.x-l.x), abs(pos.x-(l.x+l.w)))
                dyc = 0 if l.y <= pos.y <= l.y+l.h else min(abs(pos.y-l.y), abs(pos.y-(l.y+l.h)))
                if dyc < dy: closest, dx, dy = l, dxc, dyc
                elif dyc == dy and dxc < dx: closest, dx = l, dxc
            if closest is None: self.move("start", selection)
            else:
                idx = closest.start + closest.xs.index(min(closest.xs, key=lambda x: abs(x-pos.x)))
                self.update(idx, selection=selection, allow_drift=False, up=idx==closest.end)
        elif d == "right":
            if self.selection and not selection: self.update(self.selection.end, up=self.selection.endup)
            else:
                if self.idx in self.frame.wraps and self.idx==self.node.end: self.update(self.idx, selection=selection)
                elif self.idx+1 in self.frame.wraps: self.update(self.idx+1, selection=selection, up=True)
                else: self.update(self.idx + 1, selection=selection)
        elif d == "down":
            pos = vec2(self.pos.x if self.x is None else self.x, self.pos.y)
            lines = (n for n in walk(self.frame) if n.k is K.LINE and n.y > self.node.y)
            closest, dx, dy = None, math.inf, math.inf
            for l in lines:
                dxc = 0 if l.x <= pos.x <= l.x+l.w else min(abs(pos.x-l.x), abs(pos.x-(l.x+l.w)))
                dyc = 0 if l.y <= pos.y <= l.y+l.h else min(abs(pos.y-l.y), abs(pos.y-(l.y+l.h)))
                if dyc < dy: closest, dx, dy = l, dxc, dyc
                elif dyc == dy and dxc < dx: closest, dx = l, dxc
            if closest is None: self.move("end", selection)
            else:
                idx = closest.start + closest.xs.index(min(closest.xs, key=lambda x: abs(x-pos.x)))
                self.update(idx, selection=selection, allow_drift=False, up=idx==closest.end)
        elif d == "left":
            if self.selection and not selection: self.update(self.selection.start, up=self.selection.startup)
            else:
                if self.idx in self.frame.wraps and not self.idx==self.node.end: self.update(self.idx, selection=selection, up=True)
                else: self.update(self.idx - 1, selection=selection)
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

def _in_edit_view(node:Node) -> bool: return False if node is None else True if node.edit is True else _in_edit_view(node.parent)
 
def _find_edit_parent(node:Node, idx:int) -> Node:
    """Finds the node that if edit=True is set, makes idx available."""
    assert node.start <= idx <= node.end, f"Invalid Arguments: {idx=} not between {node.start=} and {node.end=}"
    assert node.k is not K.TEXT, "TEXT must have children that fill its start/end range and idx should be found separately"
    if node.k is K.LINE:
        if node.parent.k is K.TEXT: return node.parent
        else: 
            start = node.parent.start <= idx < node.parent.children[0].start
            return next((n for n in (node.parent.children if start else reversed(node.parent.children)) if n.k is K.TEXT))
    else: return next((n for n in node.children if n.k is K.TEXT)) # return any text child since any will turn on edit mode for the whole parent

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
    def __bool__(self): return all((i is not None for i in (self.i0, self.up0, self.i1, self.up1)))

def typeset(text:str, x:float, y:float, width:float, fontfamily:str, fontsize:float, fontweight:int, color:Color, lineheight:float, newline_x:float=None, align="left",
            start=0) -> Tuple[Node, List[float]]:
    lines, idata, wraps = _typeset(text, x, y, width, fontfamily, fontsize, fontweight, color, lineheight, newline_x, align, start)
    TexQuadBuffer.add(idata)
    frame.wraps.extend(wraps)
    return lines

# TODO: select font more carefully. may contain symbols not in font
# something like g = next((fonŧ[g] for font in fonts if g in font))
# TODO: line_strings is creating ambiguity, sucks, remove. index into text using line start and end to actually get the text. obviously.
@functools.cache # TODO custom cache so it can also shift cached lines down
def _typeset(text:str, x:float, y:float, width:float, fontfamily:str, fontsize:float, fontweight:int, color:Color, lineheight:float, newline_x:float=None, align="left",
             start=0) -> Tuple[Node, List[float]]:
    """
    Returns
    - Node with LINE children to add to the node tree
    - Tex quad instance data for all glyphs
    """
    font, fontweight = get_font(fontfamily, fontweight) # replace fontweight value with actual used based on what's available
    cx = x # character x offset
    linepos = vec2(x, y) # top left corner of line hitbox
    if newline_x is None: newline_x = x
    assert cx <= newline_x + width
    lines, line_strings = Node(K.TEXT, None), [""]
    lstart = 0 # char index where line started
    xs = [] # cursor x positions
    wraps = [] # idx of wraps
    idx = 0
    for idx, c in enumerate(text):
        xs.append(cx)
        if c == "\n":
            lines.children.append(Node(K.LINE, lines, x=linepos.x, y=linepos.y, w=cx - linepos.x, h=lineheight, start=start+lstart, end=start + idx+1, xs=xs))
            cx, linepos, xs, lstart = newline_x, vec2(newline_x, linepos.y + lineheight), [], idx+1
            line_strings.append("")
            continue
        g = font.glyph(c, fontsize, 72) # NOTE: dpi is 72 so the font renderer does no further scaling. It uses 72 dpi baseline.
        if cx + g.advance > newline_x + width:
            wrap_idx = fwrap + 1 if (fwrap:=line_strings[-1].rfind(" ")) >= 0 else len(line_strings[-1])
            wraps.append(wrap_idx + start + lstart)
            if wrap_idx == len(line_strings[-1]): # no wrapping necessary, just cut off
                lines.children.append(Node(K.LINE, lines, x=linepos.x, y=linepos.y, w=cx - linepos.x, h=lineheight, start=start+lstart, end=start+idx, xs=xs))
                cx, linepos = newline_x, vec2(newline_x, linepos.y + lineheight)
                xs, lstart = [cx], idx
                line_strings.append("")
            else:
                line_strings.append(line_strings[-1][wrap_idx:])
                line_strings[-2] = line_strings[-2][:wrap_idx]
                lines.children.append(Node(K.LINE, lines, x=linepos.x, y=linepos.y, w=xs[wrap_idx] - linepos.x, h=lineheight, start=start+lstart,
                                end=start+lstart+wrap_idx, xs=xs[:wrap_idx+1]))
                xs, lstart = [newline_x + x0 - xs[wrap_idx] for x0 in xs[wrap_idx:]], lstart + wrap_idx
                cx, linepos = xs[-1], vec2(newline_x, linepos.y + lineheight)

        line_strings[-1] += c
        cx += g.advance

    xs.append(cx) # first position after the last character
    if line_strings[-1] != "" or text == "":
        lines.children.append(Node(K.LINE, lines, x=linepos.x, y=linepos.y, w=cx - linepos.x, h=lineheight, start=start+lstart, end=start + idx+1, xs=xs))
    if align in ["center", "right"]:
        for c in lines.children:
            shift = x+width-c.w-c.x / (2 if align == "center" else 1)
            c.x, c.xs = c.x + shift, [x0 + shift for x0 in c.xs]

    instance_data = []
    ascentpx = font.engine.hhea.ascent * fontsize / font.engine.head.unitsPerEM
    descentpx = font.engine.hhea.descent * fontsize / font.engine.head.unitsPerEM
    total = ascentpx - descentpx
    for s, l in zip(line_strings, lines.children):
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

@GLFWframebuffersizefun
def framebuffer_size_callback(window, width, height): SS.w, SS.h, SS.resized = width, height, True

@GLFWcharfun
def char_callback(window, codepoint): write(frame, chr(codepoint))

def erase(frame:Node, right=False):
    if frame.cursor.selection: frame.text = frame.text[:frame.cursor.selection.start] + frame.text[frame.cursor.selection.end:]
    else: frame.text = frame.text[:max(frame.cursor.idx - (0 if right else 1), 0)] + frame.text[(frame.cursor.idx + (1 if right else 0)):]
    markdown = parse(frame.text)
    frame.children = [markdown]
    markdown.parent = frame
    populate_render_data(frame, SS.css, reset=True)
    frame.cursor.update(frame.cursor.idx - (0 if right or frame.cursor.selection else 1))

def write(frame:Node, text:str):
    if frame.cursor.selection: frame.text = frame.text[:frame.cursor.selection.start] + text + frame.text[frame.cursor.selection.end:]
    else: frame.text = frame.text[:frame.cursor.idx] + text + frame.text[frame.cursor.idx:]
    markdown = parse(frame.text)
    frame.children = [markdown]
    markdown.parent = frame
    populate_render_data(frame, SS.css, reset=True)
    frame.cursor.update((frame.cursor.selection.start if frame.cursor.selection else frame.cursor.idx) + len(text))

@GLFWkeyfun
def key_callback(window, key:int, scancode:int, action:int, mods:int):
    if action in [GLFW_PRESS, GLFW_REPEAT]:
        selection = bool(mods & GLFW_MOD_SHIFT)
        if key == GLFW_KEY_LEFT: frame.cursor.move("left", selection)
        elif key == GLFW_KEY_RIGHT: frame.cursor.move("right", selection)
        elif key == GLFW_KEY_UP: frame.cursor.move("up", selection)
        elif key == GLFW_KEY_DOWN: frame.cursor.move("down", selection)
        if key == GLFW_KEY_BACKSPACE: erase(frame)
        if key == GLFW_KEY_DELETE: erase(frame, right=True)
        if key == GLFW_KEY_ENTER: write(frame, "\n")
        if key == GLFW_KEY_S:
            if mods & GLFW_MOD_CONTROL: # SAVE
                with open(TEXTPATH, "w") as f: f.write(frame.text)
        if key == GLFW_KEY_A and mods & GLFW_MOD_CONTROL:  # select all
            frame.cursor.selection = Selection(0, False, len(frame.text), False)
            frame.cursor.update(len(frame.text), selection=True)
        if key == GLFW_KEY_C and mods & GLFW_MOD_CONTROL and frame.cursor.selection:
            glfwSetClipboardString(window, frame.text[frame.cursor.selection.start:frame.cursor.selection.end].encode()) # copy
        if key == GLFW_KEY_V and mods & GLFW_MOD_CONTROL: write(frame, glfwGetClipboardString(window).decode()) # paste
        if key == GLFW_KEY_X and mods & GLFW_MOD_CONTROL and frame.cursor.selection: # cut
            glfwSetClipboardString(window, frame.text[frame.cursor.selection.start:frame.cursor.selection.end].encode()) # copy
            erase(frame)
        elif key == GLFW_KEY_HOME: frame.cursor.move("start", selection)
        elif key == GLFW_KEY_END: frame.cursor.move("end", selection)

@GLFWmousebuttonfun
def mouse_callback(window, button:int, action:int, mods:int):
    if button == GLFW_MOUSE_BUTTON_1 and action is GLFW_PRESS:
        x, y = ctypes.c_double(), ctypes.c_double()
        glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
        frame.cursor.update(vec2(x.value - SCENE.x, y.value - SCENE.y), selection=glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS)

@GLFWcursorposfun
def cursor_pos_callback(window, x, y):
    if glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS:
        x, y = ctypes.c_double(), ctypes.c_double()
        glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
        frame.cursor.update(vec2(x.value - SCENE.x, y.value - SCENE.y), selection=True)

@GLFWscrollfun
def scroll_callback(window, x, y): SCENE.y += y * 40

HERITABLE_STYLES = ["color", "font-family", "font-size", "line-height", "font-weight", "text-transform"]

def pseudoclass(node:Node, pseudocl:str) -> bool: return False # TODO

def selector_match(sel:Selector, node:Node) -> bool: return all((
    sel.k is node.k,
    (all((i in node.cls if node.cls else [] for i in sel.cls))) if sel.cls else True,
    (all((i in node.ids if node.ids else [] for i in sel.ids))) if sel.ids else True,
    (all((pseudoclass(node, i) for i in sel.pseudocls))) if sel.pseudocls else True,
))

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
        QuadBuffer.clear()
        TexQuadBuffer.clear()
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
        node.x, node.y = pstyle["_block"].x + style["margin-left"], pstyle["_block"].y + max(style["margin-top"] - pstyle["_margin"].y, pstyle["_margin"].y)
        style.update({"_block": (v:=vec2(node.x + style["padding-left"], node.y + style["padding-top"])), "_inline": v, "_margin": vec2(0,0)})
    else:
        style["_width"] = pstyle["_width"]
        style["_margin"] = vec2(style["margin-right"], 0)
        style["_inline"] = pstyle["_inline"] + vec2(style["margin-left"], 0)
        node.x, node.y = style["_inline"].x, style["_inline"].y
        if node.k is K.TEXT:
            y = node.y
            if node.parent.k is K.HR and not edit_view: QuadBuffer.add([node.x, node.y, 0.6, style["_width"], 1, (c:=style["color"]).r, c.g, c.b, c.a])
            if node.parent.k is K.LI and not edit_view:
                if node.parent.parent.k is K.UL:
                    y = node.y + style["font-size"] * 0.066666 # hack to approximate browser rendering. Shift without changing node.y for talled node.
                    typeset("•", node.x - 1.25 * style["font-size"], y, style["font-size"], style["font-family"], style["font-size"]*1.4,
                            style["font-weight"], style["color"], style["line-height"]*1.075, align="right", start=node.start)
                elif node.parent.parent.k is K.OL:
                    typeset(d:=f"{node.parent.digit}.", node.x-(len(d)+0.5)*style["font-size"], y, style["font-size"]*len(d), style["font-family"],
                            style["font-size"]*1.05, style["font-weight"], style["color"], style["line-height"]*1.075, align="right", start=node.start)
            t = text[node.start:node.end].upper() if style.get("text-transform") == "uppercase" else text[node.start:node.end]
            lines = typeset(t, node.x, y, style["_width"], style["font-family"], style["font-size"], style["font-weight"], style["color"], style["line-height"],
                            pstyle["_block"].x, start=node.start)
            if node.parent.k is K.A:
                    font = get_font(style["font-family"], style["font-weight"])
                    ascentpx = font.engine.hhea.ascent * style["font-size"] / font.engine.head.unitsPerEM
                    descentpx = font.engine.hhea.descent * style["font-size"] / font.engine.head.unitsPerEM
                    total = ascentpx - descentpx
                    for l in lines.children:
                        underline_pos = font.engine.fupx(font.engine.post.underlinePosition)
                        underline_y = l.y + ascentpx + (style["line-height"] - total)/2 - underline_pos
                        h = max(1, font.engine.fupx(font.engine.post.underlineThickness))
                        QuadBuffer.add([l.x, underline_y, 0.6, l.w, h, (c:=style["color"]).r, c.g, c.b, c.a])
            steal_children(lines, node)
            c = node.children[-1]
            style["_block"] = vec2(pstyle["_block"].x, c.y + c.h)
            style["_inline"] = style["_block"].copy() if t.endswith("\n") else vec2(c.x + c.w, c.y)
        else: style["_block"] = pstyle["_block"]

    # process children and if edit view, fill any gaps (omitted formatting text) before and after
    # NOTE: Formatting text is put OUTSIDE of TEXT nodes because their start/end is outside. Expanding TEXT start/end would falsify content
    for_children.update({k:v for k,v in style.items() if k in ["_block", "_inline", "_margin", "_width"]})
    edit_view_lines = [] # List[Tuple[idx, lines]]
    child_edit_view = (edit_view or any((_in_edit_view(n) for n in node.children if n.k is K.TEXT))) and node.k not in [K.SS, K.SCENE, K.FRAME, K.BODY]
    for i, child in enumerate(node.children):
        if child.k is not K.LINE:
            if child_edit_view:
                start = end = None
                if i == 0 and node.start != child.start: start, end = node.start, child.start
                elif i > 0 and node.children[i-1].end != child.start: start, end = node.children[i-1].end, child.start
                if start is not None:
                    if child.k not in INLINE_NODES: style["_inline"].x = pstyle["_block"].x
                    lines = typeset(text[start:end], for_children["_inline"].x, for_children["_inline"].y, style["_width"], style["font-family"],
                                    style["font-size"], style["font-weight"], FORMATTING_COLOR, style["line-height"], for_children["_block"].x, start=start)
                    edit_view_lines.append((i, lines.children))
                    for_children["_inline"] = vec2((c:=lines.children[-1]).x + c.w, c.y)
                    for_children["_block"] = vec2(style["_block"].x, c.y + c.h)
                
            for_children["_block"], for_children["_inline"], for_children["_margin"] = populate_render_data(child, css_rules, for_children, text)

            if child.k is K.LINE: print(child)
            if i == len(node.children) - 1 and child_edit_view and child.end != node.end:
                lines = typeset(text[child.end:node.end], for_children["_inline"].x, for_children["_inline"].y, style["_width"], style["font-family"],
                                style["font-size"], style["font-weight"], FORMATTING_COLOR, style["line-height"], for_children["_block"].x, start=child.end)
                edit_view_lines.append((i+1, lines.children))
                for_children["_inline"] = vec2((c:=lines.children[-1]).x + c.w, c.y)
                for_children["_block"] = vec2(style["_block"].x, c.y + c.h)
    for idx, lines in reversed(edit_view_lines): # insertions are delayed until here to preserve insertion idx accuracy
        for l in reversed(lines):
            l.parent = node
            node.children.insert(idx, l)

    style.update({k:v for k,v in for_children.items() if k in ["_block", "_inline", "_margin", "_width"]})

    # update _block, _inline and _margin for return
    if style["display"] == "block":
        if node.children:
            if not getattr(node, "w", None): node.w = max([c.x + c.w for c in node.children]) - node.x # applies to frame node
            node.h = (end := node.children[-1].y + node.children[-1].h + style["padding-bottom"]) - node.y
            if (c:=style.get("background-color")) and isinstance(c, Color): # ignoring "black", "red" and such
                QuadBuffer.add([node.x, node.y, 0.6, node.w, node.h, (c:=style["background-color"]).r, c.g, c.b, c.a])
            return (_block:=vec2(pstyle["_block"].x, end)), _block, vec2(0, style["margin-bottom"]) # _block, _inline, _margin
        else:
            node.h = style["padding-top"] + style["padding-bottom"]
            return (_block:=pstyle["_block"] + vec2(0, style["padding-bottom"]), _block, vec2(0, style["margin-bottom"]))
    else:
        if node.children:
            # backgroundcolor not supported
            node.w = max([c.x + c.w for c in node.children]) - node.x
            node.h = node.children[-1].y + node.children[-1].h - node.y
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

SS = Node(K.SS, None, resized=True, scrolled=False, fonts={}, glyphAtlas=None, dpi=96, title="Spiritstream", w=700, h=1800)
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

def glUnbindBuffers():
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

class _BaseQuad:
    """base unit quad used with instancing"""
    def __init__(self):
        #                top-left                 # top-right              # bottom-left            # bottom-right
        self.vertices = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        self.indices = [0, 1, 2, 1, 2, 3]
        self.VBO, self.VAO, self.EBO = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        glGenVertexArrays(1, ctypes.byref(self.VAO))
        glGenBuffers(1, ctypes.byref(self.VBO))
        glGenBuffers(1, ctypes.byref(self.EBO))
        
        glBindVertexArray(self.VAO) # must be bound before EBO
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)

        quad_vertices_ctypes = (ctypes.c_float * len(self.vertices))(*self.vertices)
        quad_indices_ctypes = (ctypes.c_uint * len(self.indices)) (*self.indices)
        # pre allocate buffers
        glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(quad_vertices_ctypes), quad_vertices_ctypes, GL_STATIC_DRAW)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(quad_indices_ctypes), quad_indices_ctypes, GL_STATIC_DRAW)

        glUnbindBuffers()

    def delete(self):
        """Free OpenGL resources. Call before program exit"""
        glDeleteBuffers(1, self.VBO)
        glDeleteBuffers(1, self.EBO)
        glDeleteVertexArrays(1, self.VAO)
        
BaseQuad = _BaseQuad()

class _TexQuadBuffer:
    """Buffer for texture quad instances"""
    def __init__(self):
        """Create instance data buffer, vertex attribute array and instance variables"""
        self.changed = False
        self.count = 0
        self.data = [] # x, y, z, size w, h, uv offset x, y, uv size w, h, color r, g, b, a
        self.stride = 13
        self.sections = {}

        stride_c = self.stride * ctypes.sizeof(ctypes.c_float)

        self.VAO = ctypes.c_uint()
        glGenVertexArrays(1, ctypes.byref(self.VAO))
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, BaseQuad.VBO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, BaseQuad.EBO)
        
        # base quad data attributes: vertex position (loc 0), texture position (loc 1)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(1)

        self.VBO = ctypes.c_uint() # instance buffer
        glGenBuffers(1, ctypes.byref(self.VBO))
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        
        # Instance attributes (locations 2+; divisor=1 for per-instance)
        # pos (loc 2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        # size (loc 3)
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        # uv offset (loc 4)
        glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(5*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)
        # uv size (loc 5)
        glVertexAttribPointer(5, 2, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(7*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(5)
        glVertexAttribDivisor(5, 1)
        # color (loc 6)
        glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(9*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(6)
        glVertexAttribDivisor(6, 1)

        glUnbindBuffers()

    def add(self, data, name=None):
        """If name is provided, add to its section and update position and length for it, else just append data to the buffer. Applied on next self.draw call"""
        assert len(data) % self.stride == 0
        if name is not None: raise NotImplementedError
        self.data.extend(data)
        self.count += len(data) // self.stride
        self.changed = True

    def clear(self, name=None):
        """Remove data associated with name, or if name is None, set count to 0. The buffer is kept until delete, to support faster rebuilding"""
        if name is not None: raise NotImplementedError
        self.data.clear()
        self.count = 0

    def draw(self, scale, offset):
        glBindTexture(GL_TEXTURE_2D, SS.glyphAtlas.texture)
        if self.changed: # upload buffer. NOTE: no mechanism shrinks the buffer if fewer quads are needed as before. same applies to quad buffer
            data_c = (ctypes.c_float * (length:=self.count*self.stride))(*self.data[:length])
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
            glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(data_c), data_c, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.changed = False
        texquadShader.use()
        texquadShader.setUniform("scale", scale, "2f") # inverted y axis. from my view (0,0) is the top left corner, like in browsers
        texquadShader.setUniform("offset", offset, "2f")
        glBindVertexArray(self.VAO)
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, self.count)
    
    def delete(self):
        """Free OpenGL resources. Call before program exit"""
        glDeleteBuffers(1, self.VBO)
        glDeleteVertexArrays(1, self.VAO)

TexQuadBuffer = _TexQuadBuffer()

@dataclass
class Segment:
    name:str
    start:int
    length:int

class _QuadBuffer:
    """Buffer for untextured quad instances"""
    def __init__(self):
        """Create instance data buffer, vertex attribute array and instance variables"""
        self.changed = False
        self.data = [] # x, y, z, size w, h, color r, g, b, a
        self.stride = 9
        self.segments:List[Segment] = []

        stride_c = self.stride * ctypes.sizeof(ctypes.c_float)

        self.VAO = ctypes.c_uint()
        glGenVertexArrays(1, ctypes.byref(self.VAO))
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, BaseQuad.VBO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, BaseQuad.EBO)
        
        # base quad data attributes: vertex position (loc 0), texture position (loc 1)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(1)

        self.VBO = ctypes.c_uint() # instance buffer
        glGenBuffers(1, ctypes.byref(self.VBO))
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)

        # Instanced attributes (locations 2+; divisor=1 for per-instance)
        # pos (loc 2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        # size (loc 3)
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        # color (loc 4)
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, stride_c, ctypes.c_void_p(5*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)

        glUnbindBuffers()

    @property
    def count(self): return sum((s.length for s in self.segments)) // self.stride

    def add(self, data, name=None):
        """If name is provided, add to its section and update position and length for it, else just append data to the buffer. Applied on next self.draw call"""
        assert len(data) % self.stride == 0
        if name is None: name = "generic"
        if (segment:=list(filter(lambda s: s.name == name, self.segments))) == []:
            if self.segments: self.segments.append(Segment(name, self.segments[-1].start + self.segments[-1].length, len(data)))
            else: self.segments.append(Segment(name, 0, len(data)))
            self.data.extend(data)
        else:
            segment = segment[0]
            for s in self.segments:
                if s.start > segment.start: s.start += len(data)
            self.data = self.data[:segment.start + segment.length] + data + self.data[segment.start + segment.length:]
            segment.length += len(data)
        self.changed = True
    
    def replace(self, data, name):
        if (segment:=list(filter(lambda s: s.name == name, self.segments))) == []: self.add(data, name)
        else: 
            segment = segment[0]
            shift = len(data) - segment.length
            if shift != 0:
                for s in self.segments:
                    if s is not segment and s.start >= segment.start: s.start += shift
            self.data[segment.start:segment.start+segment.length] = data
            segment.length = len(data)
            self.changed = True

    def clear(self, name=None):
        """Remove data associated with name, or if name is None, set count to 0. The buffer is kept until delete, to support faster rebuilding"""
        if name is not None: raise NotImplementedError
        self.data.clear()
        self.segments.clear()

    def draw(self, scale, offset):
        if self.changed: # upload buffer
            data_c = (ctypes.c_float * (length:=self.count*self.stride))(*self.data[:length])
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
            glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(data_c), data_c, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.changed = False
        quadShader.use()
        quadShader.setUniform("scale", scale, "2f") # inverted y axis. from my view (0,0) is the top left corner, like in browsers
        quadShader.setUniform("offset", offset, "2f")
        glBindVertexArray(self.VAO)
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, self.count)

    def delete(self):
        """Free OpenGL resources. Call before program exit"""
        glDeleteBuffers(1, self.VBO)
        glDeleteVertexArrays(1, self.VAO)

QuadBuffer = _QuadBuffer()

texquadShader = Shader(p:=Path(__file__).parent / "spiritstream/shaders/texquad.vert", p.parent / "texquad.frag", ["glyphAtlas", "scale", "offset"])
texquadShader.setUniform("glyphAtlas", 0, "1i")  # 0 means GL_TEXTURE0)
quadShader = Shader(p:=Path(__file__).parent / "spiritstream/shaders/quad.vert", p.parent / "quad.frag", ["scale", "offset"])

glEnable(GL_DEPTH_TEST)
glClearDepth(1)
glClearColor(0, 0, 0, 1)

TEXTPATH.touch()
with open(TEXTPATH, "r") as f: markdown = parse(t:=f.read())
with open(CSSPATH, "r") as f: SS.css, SS.fonts = parseCSS(f.read())
default_css = {
    Selector(K.BODY):{"font-size": (16, "px"), "font-weight": 400},
    Selector(K.B):{"font-weight": 700}
}
for rule, declarations in default_css.items():
    for k,v in declarations.items(): SS.css.setdefault(rule, declarations).setdefault(k, v)
frame = Node(K.FRAME, SCENE, [markdown], text=t, editnodes=set(), wraps=[], edit=False)
markdown.parent = frame

# TODO: load dynamically when used
for font in SS.fonts:
    assert font["format"] == "truetype"
    font["Font"] = Font(Path(CSSPATH).parent.joinpath(font["url"]).resolve())

# t0 = time.time()
populate_render_data(frame, SS.css, reset=True)
# print(f"{time.time() - t0:.3f}")

from spiritstream.tree import serialize
with open("./out.html", "w") as f: f.write(serialize(frame, CSSPATH))

frame.cursor = Cursor(frame, idx=CURSORIDX, up=CURSORUP) # after rendertree so it can find line elements in the tree
SCENE.children = [frame]

# show(SS)

while not glfwWindowShouldClose(window):
    if glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS: glfwSetWindowShouldClose(window, GLFW_TRUE)

    if SS.resized:
        SCENE.x = max(0, (SS.w - frame.w)/2)
        SS.resized = False
        glViewport(0, 0, SS.w, SS.h)
        populate_render_data(frame, SS.css, reset=True)
        frame.cursor.update(frame.cursor.idx, up=frame.cursor.idx==frame.cursor.node.end, selection=frame.cursor.selection)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    scale, offset = (2 / SS.w, -2 / SS.h), tuple(vec2(-1 + SCENE.x * 2 / SS.w, 1 + SCENE.y * -2 / SS.h).components())

    QuadBuffer.draw(scale, offset)
    TexQuadBuffer.draw(scale, offset)
    
    if (error:=glGetError()) != GL_NO_ERROR: print(f"OpenGL Error: {hex(error)}")

    glfwSwapBuffers(window)
    glfwWaitEvents()

with open(WORKSPACEPATH, "w") as f: json.dump({"textpath": TEXTPATH.as_posix(), "sceney": SCENE.y, "cursoridx":frame.cursor.idx,
                                               "cursorup": frame.cursor.idx == frame.cursor.node.end}, f)

if SAVE_GLYPHATLAS:
    from spiritstream.image import Image
    Image.write(list(reversed(SS.glyphAtlas.bitmap)), Path(__file__).parent / "GlyphAtlas.bmp")

if CACHE_GLYPHATLAS:
    with open(atlaspath, "wb") as f: pickle.dump(SS.glyphAtlas, f)

BaseQuad.delete()
QuadBuffer.delete()
TexQuadBuffer.delete()
texquadShader.delete()
quadShader.delete()
glfwTerminate()