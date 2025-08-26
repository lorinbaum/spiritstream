from pprint import pprint
import math, time, pickle
from enum import Enum, auto
from typing import Union, List, Dict, Tuple
from spiritstream.vec import vec2
from spiritstream.bindings.glfw import *
from spiritstream.bindings.opengl import *
from spiritstream.font import Font, Glyph
from dataclasses import dataclass
from spiritstream.shader import Shader
from spiritstream.textureatlas import TextureAtlas
from spiritstream.helpers import PRINT_TIMINGS, CACHE_GLYPHATLAS, SAVE_GLYPHATLAS
from spiritstream.tree import parse, parseCSS, Node, K, Color, color_from_hex, walk, show, INLINE_NODES

if PRINT_TIMINGS: FIRSTTIME = LASTTIME = time.time()

TEXTPATH = "text.txt"
CSSPATH = "test/test.css"
EDIT_VIEW = True
FORMATTING_COLOR = color_from_hex(0xaaa)

"""
selection quads per node (could try and merge)
cursor positioning rewrite, now navigates the tree (slightly slow because some nodes don't display, only their children).
Each node has lines with newline flag, y, and cursorcoords.
Source view also does formatting if markdown and displays markdown in different color. This avoid weird shifting when clicking on stuff.

OPTIONAL? Node also stores which part of the text it covers.

Node
    kind:Kinds
    parent:Node
    children:List[Node]
    data:Dict[]

@dataclass(frozen=True)
class Line:
    x:float
    y:float
    w:float
    h:float
    start:int
    end:int
    newline:bool
    cursorxs:Tuple

OPTIMIZATION: cursor pos search could first check up the tree if the new position is in the same parent, then descend down the other branch if necessary.

def text(text:str, x:float, y:float, width:float, font:Font, fontsize:float, color:Color, lineheight:float, newline_x:float=None) -> List[Line]:

text edit trigger rerender. edit only happens in raw text or piece table. text stored in frame node.
def tree_update(head:Node, cursor:Cursor, action:int, text)

framenode stores its cursors. only one cursor for now.
class Cursor:
    frame: framehead
"""


class Cursor:
    def __init__(self, frame:Node):
        self.frame = frame # keep reference for cursor_coords and linewraps
        self.idx = 0
        self.pos = vec2()
        self.x = None
        # self.left = None
        # self.rpos = vec2() # relative to self.text
        # self._apos = vec2() # absolute. derived from self.text.x, self.text.y and rpos. don't set directly or derive from self.rpos.
        # self.idx = None
        # self.line = None
        # self.x = 0 # Moving up and down using arrow keys can lead to drifting left or right. To avoid, stores original x in this variable.

        # self.update(0)

    def update(self, pos:Union[int, vec2], allow_drift=True, left=False):
        assert isinstance(left, bool) and isinstance(allow_drift, bool)
        assert isinstance(pos, (int, vec2)), f"Wrong argument type \"pos\": {type(pos)}. Can only use integer (index of char in the text) or vec2 (screen coordinate)"
        cursorcoords = self.text.cursorcoords
        if isinstance(pos, vec2): # get idx from pos
            idx = 0
            if not allow_drift: pos.x = self.x
            if pos.y > self.text.cursorcoords[-1].y + (self.text.lineheightUsed) / 2: idx, allow_drift = len(self.text.cursorcoords) - 1, True
            elif pos.y < self.text.cursorcoords[0].y - (self.text.lineheightUsed) / 2: idx, allow_drift = 0, True
            else:
                closest_x = closest_y = math.inf
                for i, l in enumerate(self.text.visible_lines):
                    if (dy:=abs(pos.y - l.y)) < closest_y: closest_y, self.line = dy, l
                    else: break
                for i, c in enumerate([vec2(-2.01, None)] + self.text.cursorcoords[self.line.start+1:self.line.end+1]):
                    if (dx:=abs(pos.x - c.x)) < closest_x:
                        closest_x, idx, left = dx, self.line.start + i, i == 0 and not self.line.newline
                    else: break
        else:
            idx = pos % len(cursorcoords)
            self.line = next((l for l in self.text.lines if l.start <= idx <= l.end))
        if idx == self.idx: return
        self.idx = idx
        self.left = left
        self.rpos = vec2(-2, cursorcoords[(self.idx + 1) % len(cursorcoords)].y) if left else cursorcoords[self.idx] # relative
        self._apos = self.rpos + vec2(self.text.x, self.text.y) # absolute
        
        # scroll into view
        if self._apos.y - self.text.lineheightUsed < Scene.y:
            Scene.y = self._apos.y - self.text.lineheightUsed
            Scene.scrolled = True
        elif self._apos.y > Scene.y + Scene.h:
            Scene.y = self._apos.y - Scene.h
            Scene.scrolled = True

        h = self.text.lineheightUsed
        #                           x,y                                    z    w   h  r    g    b    a
        quad_instance_data[:9] = [*(self._apos + vec2(0, 3)).components(), 0.2, 2, -h, 1.0, 1.0, 1.0, 1.0]
        global quads_changed
        quads_changed = True

        if allow_drift: self.x = self.rpos.x

# class Selection:
#     def __init__(self, text:"Text"):
#         self.color = color_from_hex(0x423024)
#         self.text = text
#         self.start = self.startleft = self.end = self.endleft = self.pos1 = self.pos1left = None
#         self.instance_count = 0
    
#     def update(self, pos1=None, left1=None, pos2=None, left2=None):
#         self.instance_count = 0
#         done = False # used for existing the for loop below
#         if pos1 == left1 == pos2 == left2 == None: # all are set or none are set. this allows calling selection.update() to rerender it
#             assert all([i != None for i in [self.pos1, self.pos1left, self.start, self.startleft, self.end, self.endleft]])
#             pos1, left1, pos2, left2 = self.start, self.startleft, self.end, self.endleft
#         else:
#             assert all([i != None for i in [pos1, left1, pos2, left2]])
#             if self.pos1 == None: self.pos1, self.pos1left = pos1, left1
#             self.start, self.startleft, self.end, self.endleft = (self.pos1, left1, pos2, left2) if self.pos1 < pos2 else (pos2, left2, self.pos1, left1)
#             if self.start == self.end: return self.reset()
#         for i, l in enumerate((l for l in self.text.visible_lines if l.end >= self.start)):
#             if i == 0 and l.start <= self.start:
#                 if self.startleft and l.end == self.start: continue
#                 x, y = self.text.cursorcoords[self.start].components()
#             else: x, y = 0, l.y
#             if self.end <= l.end: w, done = self.text.cursorcoords[self.end].x - x, True 
#             else: w = self.text.cursorcoords[l.end].x - x + 5
#             instance_data = [x+self.text.x, y+self.text.y+3, 0.9, w, -self.text.lineheightUsed, self.color.r, self.color.g, self.color.b, self.color.a]
#             # +1 offset because first instance is cursor
#             quad_instance_data[(self.instance_count+1)*(s:=len(instance_data)):(self.instance_count+2)*s] = instance_data
#             self.instance_count += 1
#             if done: break
#         global quads_changed
#         quads_changed = True

#     def reset(self):
#         self.instance_count = 0
#         self.start = self.end = self.pos1 = self.startleft = self.pos1left = self.endleft = None

@dataclass(frozen=True)
class Line:
    x:float
    y:float
    w:float
    h:float
    start:int
    end:int
    newline:bool
    cursorxs:Tuple

    def __repr__(self): return f"\033[93mLine\033[0m(\033[94mx=\033[0m{self.x}, \033[94my=\033[0m{self.y}, \033[94mw=\033[0m{self.w}," \
        f"\033[94mh=\033[0m{self.h}, \033[94mstart=\033[0m{self.start}, \033[94mend=\033[0m{self.end}, \033[94mnewline=\033[0m{self.newline})"

# TODO: select font more carefully. may contain symbols not in font
# something like g = next((fonŧ[g] for font in fonts if g in font))
def text(text:str, x:float, y:float, width:float, font:str, fontsize:float, color:Color, lineheight:float, newline_x:float=None, align="left", start=0) -> List[Line]:
    """Returns LINE nodes to add to the node tree and populates quad buffers to draw the text"""
    cx = x # character x offset
    linepos = vec2(x, y) # top left corner of line hitbox
    if newline_x is None: newline_x = x
    assert cx <= newline_x + width
    ret, newline, line_strings = [], True, [""]
    lstart = 0 # char index where line started
    xs = [] # cursor x positions
    for idx, c in enumerate(text):
        xs.append(cx)
        if c == "\n":
            ret.append(Line(linepos.x, linepos.y, cx - linepos.x, lineheight, start+lstart, start + idx+1, newline, xs))
            cx, linepos = newline_x, vec2(newline_x, linepos.y + lineheight)
            newline, xs, lstart = True, [], idx+1
            line_strings.append("")
            continue
        g = SS.fonts[font].glyph(c, fontsize, 72) # NOTE: dpi is 72 so the font renderer does no further scaling. It uses 72 dpi baseline.
        if cx + g.advance > newline_x + width:
            wrap_idx = fwrap + 1 if (fwrap:=line_strings[-1].rfind(" ")) >= 0 else len(line_strings[-1])
            if wrap_idx == len(line_strings[-1]): # no wrapping necessary, just cut off
                ret.append(Line(linepos.x, linepos.y, cx - linepos.x, lineheight, start+lstart, start+idx+1, newline, xs))
                cx, linepos = newline_x, vec2(newline_x, linepos.y + lineheight)
                newline, xs, lstart = False, [cx], idx+1
                line_strings.append("")
            else:
                line_strings.append(line_strings[-1][wrap_idx:])
                line_strings[-2] = line_strings[-2][:wrap_idx]
                ret.append(Line(linepos.x, linepos.y, xs[wrap_idx] - linepos.x, lineheight, start+lstart, start + lstart + wrap_idx, newline, xs[:wrap_idx+1]))
                newline, xs, lstart = False, [newline_x + x0 - xs[wrap_idx] for x0 in xs[wrap_idx:]], lstart + wrap_idx # TODO: test +1 or not
                cx, linepos = xs[-1], vec2(newline_x, linepos.y + lineheight)

        line_strings[-1] += c
        cx += g.advance

    xs.append(cx) # first position after the last character
    if line_strings[-1] != "": ret.append(Line(*linepos.components(), cx - linepos.x, lineheight, start+lstart, start + idx+1, newline, xs))
    if align == "right": ret = [Line(l.x+(shift:=x+width-l.w-l.x),l.y,l.w,l.h,l.start,l.end,l.newline,[c+shift for c in l.cursorxs]) for l in ret]
    elif align == "center": ret = [Line(l.x+(shift:=(x+width-l.w-l.x)/2),l.y,l.w,l.h,l.start,l.end,l.newline,[c+shift for c in l.cursorxs]) for l in ret]

    global tex_quad_instance_count, tex_quad_instance_stride, tex_quads_changed
    ascentpx = SS.fonts[font].engine.hhea.ascent * fontsize / SS.fonts[font].engine.head.unitsPerEM
    descentpx = SS.fonts[font].engine.hhea.descent * fontsize / SS.fonts[font].engine.head.unitsPerEM
    total = ascentpx - descentpx
    for s, l in zip(line_strings, ret):
        for i, c in enumerate(s):
            g = SS.fonts[font].glyph(c, fontsize, 72)
            if c != " ":
                key = f"{font}_{ord(c)}_{fontsize}"
                instance_data = [l.cursorxs[i] + g.bearing.x, l.y + ascentpx + (lineheight - total)/2 + (g.size.y - g.bearing.y), 0.5, # pos
                    g.size.x, -g.size.y, # size
                    *(SS.glyphAtlas.coordinates[key] if key in SS.glyphAtlas.coordinates else SS.glyphAtlas.add(key, SS.fonts[font].render(c, fontsize, 72))), # uv offset and size
                    color.r, color.g, color.b, color.a] # color 
                tex_quad_instance_data[(count:=tex_quad_instance_count)*(stride:=tex_quad_instance_stride):(count+1)*stride] = instance_data
                tex_quad_instance_count, tex_quads_changed = tex_quad_instance_count + 1, True
    return ret


# class Text:
#     def __init__(self, text:str, x, y, w, h, color):
#         self.text = text
#         self.x = x
#         self.y = y
#         self.w = w
#         self.h = h
#         self.color = color
#         self.font = Scene.font
#         self.fontsize = Scene.fontsize
#         self.lineheight = Scene.lineheight
#         self.lineheightUsed = self.fontsize * self.lineheight
#         self.lines:List[Line] # useful for determining which portion of the text is currently visible, cursor movement and editing relating to lines
#         self.cursorcoords:List[vec2]
#         self.instance_count = 0 # number of quad instances. Used for rendering them.

#         self.update()
#         self.cursor = Cursor(self)

#         self.selection = Selection(self)
#         Scene.nodes.append(self) # is this a good idea?

#     def goto(self, pos:Union[int, vec2], allow_drift=True, left=False, selection=False):
#         oldidx, oldleft  = self.cursor.idx, self.cursor.left
#         self.cursor.update(pos, allow_drift=allow_drift, left=left)
#         if selection: self.selection.update(oldidx, oldleft, self.cursor.idx, self.cursor.left)
#         else: self.selection.reset()

#     @property
#     def visible_lines(self): return (l for l in self.lines if Scene.y + Scene.h + self.lineheightUsed >= l.y > self.y - self.lineheightUsed)

#     def update(self):
#         if PRINT_TIMINGS:
#             LASTTIME = time.time()
#             GLYPHLOADTIME = 0
#             GLYPHRENDERTIME = 0
#             INSTANCEUPDATETIME = 0
#             TIMING_NEWGLYPHCOUNT = 0
#         lines = []
#         cursorcoords = []
#         offset = vec2(0, self.lineheightUsed) # bottom left corner of character in the first line
#         newline = True
#         instance_count = 0
#         for i, char in enumerate(self.text):
#             cursorcoords.append(offset.copy())
#             if ord(char) == 10:  # newline
#                 lines.append(Line(newline, offset.y, 0 if len(lines) == 0 else lines[-1].end + (1 if newline else 0), i))
#                 newline = True
#                 offset = vec2(0, offset.y + self.lineheightUsed)
#                 continue
#             if PRINT_TIMINGS: PREGLYPHLOADTIME = time.time()
#             g = Scene.fonts[self.font].glyph(char, self.fontsize, Scene.dpi)
#             if PRINT_TIMINGS: GLYPHLOADTIME += time.time() - PREGLYPHLOADTIME
#             if offset.x + g.advance > self.w:
#                 assert i > 0
#                 lines.append(Line(newline, offset.y, 0 if len(lines) == 0 else lines[-1].end + (1 if newline else 0), i))
#                 newline = False
#                 offset = vec2(0, offset.y + self.lineheightUsed) # new line, no word splitting
#             if ord(char) == 32: # space
#                 offset.x += g.advance
#                 continue
#             k = Scene.font + char + str(self.fontsize)
#             if PRINT_TIMINGS: PREGLYPHRENDERTIME, TIMING_NEWGLYPH = time.time(), k not in Scene.glyphAtlas.coordinates
#             u, v, w, h = Scene.glyphAtlas.coordinates[k] if k in Scene.glyphAtlas.coordinates else Scene.glyphAtlas.add(k, Scene.fonts[self.font].render(char, self.fontsize, Scene.dpi))
#             if PRINT_TIMINGS:
#                 if TIMING_NEWGLYPH:
#                     GLYPHRENDERTIME += time.time() - PREGLYPHRENDERTIME
#                     TIMING_NEWGLYPHCOUNT += 1
#                 PREINSTANCEUPDATETIME = time.time()
                
#             instance_data = [
#                 self.x + offset.x + g.bearing.x, self.y + offset.y + (g.size.y - g.bearing.y), 0.5, # pos
#                 g.size.x, -g.size.y, # size
#                 u, v, # uv offset
#                 w, h, # uv size
#                 self.color.r, self.color.g, self.color.b, self.color.a # color
#             ]
#             tex_quad_instance_data[instance_count*(s:=len(instance_data)):(instance_count+1)*s] = instance_data
#             global tex_quads_changed
#             tex_quads_changed = True
#             instance_count += 1
#             offset.x += g.advance
#             if PRINT_TIMINGS: INSTANCEUPDATETIME += time.time() - PREINSTANCEUPDATETIME

#         lines.append(Line(newline, offset.y, lines[-1].end + (1 if newline else 0) if len(lines) > 0 else 0, len(cursorcoords)))
#         cursorcoords.append((offset-vec2(2, 0)).copy()) # first position after the last character
#         self.lines, self.cursorcoords, self.instance_count = lines, cursorcoords, instance_count
#         if PRINT_TIMINGS:
#             print(f"{time.time() - LASTTIME:.3f}: Total text update:")
#             print(f"  {GLYPHLOADTIME:.3f}: Total glyph loading")
#             if TIMING_NEWGLYPHCOUNT: print(f"  {GLYPHRENDERTIME:.3f}: Total new glyph rendering ({TIMING_NEWGLYPHCOUNT} glyphs, {GLYPHRENDERTIME / TIMING_NEWGLYPHCOUNT:.3f} average per glyph)")
#             print(f"  {INSTANCEUPDATETIME:.3f}: Total instance data update\n      ")

#     def write(self, t:Union[int, str]):
#         if t == None: return # can happen on empty clipboard
#         if isinstance(t, int): t = chr(t) # codepoint
#         if self.selection.instance_count == 0:
#             self.text = self.text[:self.cursor.idx] + t + self.text[self.cursor.idx:]
#             self.update()
#             self.cursor.update(self.cursor.idx + len(t))
#         else:
#             self.text = self.text[:self.selection.start] + t + self.text[self.selection.end:]
#             self.update()
#             self.cursor.update(self.selection.start + len(t))
#             self.selection.reset()

#     def erase(self, right=False):
#         if self.selection.instance_count == 0:
#             if self.cursor.idx == 0 and right == False: return
#             offset = 1 if right else 0
#             self.text = self.text[:self.cursor.idx-1 + offset] + self.text[self.cursor.idx + offset:]
#             self.update()
#             self.cursor.update(self.cursor.idx - 1 + offset)
#         else:
#             self.text = self.text[:self.selection.start] + self.text[self.selection.end:]
#             self.update()
#             self.cursor.update(self.selection.start)
#             self.selection.reset()

# @GLFWframebuffersizefun
# def framebuffer_size_callback(window, width, height):
#     Scene.w, Scene.h, Scene.resized = width, height, True

# @GLFWcharfun
# def char_callback(window, codepoint): text.write(codepoint)

# @GLFWkeyfun
# def key_callback(window, key:int, scancode:int, action:int, mods:int):
#     if action in [GLFW_PRESS, GLFW_REPEAT]:
#         selection = bool(mods & GLFW_MOD_SHIFT)
#         if key == GLFW_KEY_LEFT: text.goto(text.selection.start) if text.selection.start != None and not selection else text.goto(text.cursor.idx - 1, left=text.cursor.idx - 1 == text.cursor.line.start and not text.cursor.line.newline, selection=selection)
#         if key == GLFW_KEY_RIGHT: text.goto(text.selection.end) if text.selection.end != None and not selection else text.goto(text.cursor.idx+1, selection=selection)
#         if key == GLFW_KEY_UP: text.goto(text.cursor.rpos - vec2(0, text.lineheightUsed), allow_drift=False, selection=selection)
#         if key == GLFW_KEY_DOWN: text.goto(text.cursor.rpos + vec2(0, text.lineheightUsed), allow_drift=False, selection=selection)
#         if key == GLFW_KEY_BACKSPACE: text.erase()
#         if key == GLFW_KEY_DELETE: text.erase(right=True)
#         if key == GLFW_KEY_ENTER: text.write(ord("\n"))
#         if key == GLFW_KEY_S:
#             if mods & GLFW_MOD_CONTROL: # SAVE
#                 with open("text.txt", "w") as f: f.write(text.text)
#         if key == GLFW_KEY_A and mods & GLFW_MOD_CONTROL:  # select all
#             text.selection.reset()
#             text.goto(0)
#             text.goto(len(text.text), selection=True)
#         if key == GLFW_KEY_C and mods & GLFW_MOD_CONTROL and text.selection.instance_count > 0: glfwSetClipboardString(window, text.text[text.selection.start:text.selection.end].encode()) # copy
#         if key == GLFW_KEY_V and mods & GLFW_MOD_CONTROL: text.write(glfwGetClipboardString(window).decode()) # paste
#         if key == GLFW_KEY_X and mods & GLFW_MOD_CONTROL and text.selection.instance_count > 0: # cut
#             glfwSetClipboardString(window, text.text[text.selection.start:text.selection.end].encode()) # copy
#             text.erase()
#         if key == GLFW_KEY_HOME: text.goto(text.cursor.line.start, allow_drift=True, left=True, selection=selection)
#         if key == GLFW_KEY_END: text.goto(text.cursor.line.end, allow_drift=True, selection=selection) # TODO: double pressing home in wrapping lines, works, but not double pressing end

# @GLFWmousebuttonfun
# def mouse_callback(window, button:int, action:int, mods:int):
#     if button == GLFW_MOUSE_BUTTON_1 and action in [GLFW_PRESS, GLFW_RELEASE]:
#         x, y = ctypes.c_double(), ctypes.c_double()
#         glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
#         text.goto(vec2(x.value - text.x, y.value + text.y), selection=action == GLFW_RELEASE or glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) # adjust for offset vector

# @GLFWcursorposfun
# def cursor_pos_callback(window, x, y):
#     if glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS:
#         x, y = ctypes.c_double(), ctypes.c_double()
#         glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
#         text.goto(vec2(x.value - text.x, y.value + text.y), selection=True) # adjust for offset vector

# @GLFWscrollfun
# def scroll_callback(window, x, y):
#     Scene.y -= y * 40
#     Scene.scrolled = True



HERITABLE_STYLES = ["color", "font-family", "font-size", "line-height", "font-weight"]

def render_tree(node:Node, css_rules:Dict, pstyle:Dict=None, _frame:Node=None) -> Tuple[vec2, vec2, vec2]:
    """
    Recusrively draws nodes and text in the tree according to the style information in css_rules.
    pstyle (parent style) holds information specific to a node's children, like inherited properties and below's custom properties.
    4 internal properties are added to styling information to help with positioning:
        _block:vec2     absolute position of next child / sibling if block element: child if passed pstyle and sibling if returned to the parent
        _inline:vec2    same for inline elements
        _margin:vec2    Used for margin collapse. _margin.x = margin-right of latest inline element, _margin.y = margin-bottom of latest block element
        _width:float    width of content box, passed to children
    populates x, y, w, h properties of each node and adds Line children to each node that has content
    _frame keeps a reference to the frame node that stores the text I want to render.
    """
    if node.k is K.FRAME: _frame = node
    # inherit and apply style
    if pstyle is None: pstyle = {"font-size": (16.0, "px")}
    for k, v in {"_block": vec2(0, 0), "_inline": vec2(0, 0), "_margin": vec2(0, 0), "_width": 0.0}.items(): pstyle.setdefault(k, v)
    assert all([isinstance(v, vec2) for v in [pstyle["_block"], pstyle["_inline"], pstyle["_margin"]]]) and isinstance(pstyle["_width"], float)
    style = {k:v for k, v in pstyle.items() if k in HERITABLE_STYLES}
    style.update({k:v for sel, rules in css_rules.items() for k, v in rules.items() if sel == "*"})
    style.update({k:v for sel, rules in css_rules.items() for k, v in rules.items() if sel == node.k.name.lower()}) # TODO: sel like p code
    style.update({k:csspx(pstyle, style, k) for k in ["width", "font-size"] if k in style}) # update first, other relative units need them
    for_children = {k:v for k,v in style.items() if k in HERITABLE_STYLES} # pass on before more conversion so relative values are recomputed
    style = {k:csspx(pstyle, style, k) for k in style.keys()} # convert px, em, %, auto to device pixels

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
            if node.parent.k is K.LI and not EDIT_VIEW:
                if node.parent.parent.k is K.UL:
                    y = node.y + style["font-size"] * 0.066666 # hack to approximate browser rendering. Shift without changing node.y for talled node.
                    text("•", node.x - 1.25 * style["font-size"], y, style["font-size"], style["font-family"], style["font-size"]*1.4, style["color"],
                         style["line-height"]*1.075, align="right", start=node.start)
                elif node.parent.parent.k is K.OL:
                    text(d:=f"{node.parent.digit}.", node.x - (len(d)+0.5) * style["font-size"], y, style["font-size"] * len(d), style["font-family"],
                         style["font-size"]*1.05, style["color"], style["line-height"]*1.075, align="right", start=node.start)
            t = _frame.text[node.start:node.end].upper() if style.get("text-transform") == "uppercase" else _frame.text[node.start:node.end]
            node.children = text(t, node.x, y, style["_width"], style["font-family"], style["font-size"], style["color"], style["line-height"],
                                 pstyle["_block"].x, start=node.start)
            c = node.children[-1]
            style["_block"] = vec2(pstyle["_block"].x, c.y + c.h)
            style["_inline"] = style["_block"].copy() if t.endswith("\n") else vec2(c.x + c.w, c.y)
    
    # In EDIT_VIEW insert Lines if first child start later than this node to include formatting text
    if EDIT_VIEW and node.k not in [K.SS, K.SCENE, K.FRAME, K.BODY]:
        t = None 
        if node.children and isinstance(node.children[0], Node) and node.children[0].start > node.start: t = _frame.text[node.start:node.children[0].start]
        elif not node.children and node.start-node.end != 0: t = _frame.text[node.start:node.end] # node with formatting but no text child
        if t:
            if node.k not in INLINE_NODES: style["_inline"] = style["_block"].copy()
            lines = text(t, style["_inline"].x, style["_inline"].y, style["_width"], style["font-family"], style["font-size"], FORMATTING_COLOR,
                         style["line-height"], pstyle["_block"].x, start=node.start)
            for line in reversed(lines): node.children.insert(0, line)
            style["_inline"] = vec2(lines[-1].x + lines[-1].w, lines[-1].y)
            style["_block"] = vec2(pstyle["_block"].x, lines[-1].y + lines[-1].h)

    # process children
    for_children.update({k:v for k,v in style.items() if k in ["_block", "_inline", "_margin", "_width"]})
    for child in node.children:
        if isinstance(child, Node): for_children["_block"], for_children["_inline"], for_children["_margin"] = render_tree(child, css_rules, for_children, _frame)
    style.update({k:v for k,v in for_children.items() if k in ["_block", "_inline", "_margin", "_width"]})
    
    # In EDIT_VIEW append lines if this is the last child and ends before its parent to include formatting text
    if EDIT_VIEW and node.k not in [K.SS, K.SCENE, K.FRAME, K.BODY] and node.parent.children[-1] is node and node.end < node.parent.end:
        node.parent.children.extend(text(_frame.text[node.end:node.parent.end], style["_inline"].x, style["_inline"].y, style["_width"],style["font-family"],
                                         style["font-size"], FORMATTING_COLOR, style["line-height"], pstyle["_block"].x, start=node.end))
        style["_inline"] = vec2((c:=node.parent.children[-1]).x + c.w, c.y)
        style["_block"] = vec2(pstyle["_block"].x, c.y + c.h)

    # update _block, _inline and _margin for return
    if style["display"] == "block":
        if node.children:
            node.h = (end := node.children[-1].y + node.children[-1].h + style["padding-bottom"]) - node.y
            if (c:=style.get("background-color")) and isinstance(c, Color): # ignoring "black", "red" and such
                global quads_changed, quad_instance_count
                quad_instance_data[quad_instance_count*9:(quad_instance_count+1)*9] = [node.x, node.y, 0.6, node.w, node.h, (c:=style["background-color"]).r, c.g, c.b, c.a]
                quads_changed, quad_instance_count = True, quad_instance_count + 1
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
        elif u == "%": return pstyle["_width"] * v / 100
    return v[0] if isinstance(v, tuple) and len(v) == 1 else v

SS = Node(K.SS, None, resized=True, scrolled=False, fonts={}, glyphAtlas=None, dpi=96, title="Spiritstream", w=700, h=1600)
SCENE = Node(K.SCENE, SS, x=0, y=0)
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

# glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)
# glfwSetCharCallback(window, char_callback)
# glfwSetKeyCallback(window, key_callback)
# glfwSetMouseButtonCallback(window, mouse_callback)
# glfwSetCursorPosCallback(window, cursor_pos_callback)
# glfwSetScrollCallback(window, scroll_callback)

# base unit quad used with instancing
#                top-left                 # top-right              # bottom-left            # bottom-right
quad_vertices = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
quad_indices = [0, 1, 2, 1, 2, 3]
quad_VBO, tex_quad_VAO, quad_EBO = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
glGenBuffers(1, ctypes.byref(quad_VBO))
glGenVertexArrays(1, ctypes.byref(tex_quad_VAO))
glGenBuffers(1, ctypes.byref(quad_EBO))
glBindBuffer(GL_ARRAY_BUFFER, quad_VBO)
glBindVertexArray(tex_quad_VAO) # must be bound before EBO
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quad_EBO)
quad_vertices_ctypes = (ctypes.c_float * len(quad_vertices))(*quad_vertices)
quad_indices_ctypes = (ctypes.c_uint * len(quad_indices)) (*quad_indices)
glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(quad_vertices_ctypes), quad_vertices_ctypes, GL_STATIC_DRAW) # pre allocate buffer
glBufferData(GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(quad_indices_ctypes), quad_indices_ctypes, GL_STATIC_DRAW) # pre allocate buffer

# VAO for quads with textures
# position (loc 0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
glEnableVertexAttribArray(0)
# tex (loc 1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(1)
# quad instances
tex_quads_changed = False
tex_quad_instance_count = 0
tex_quad_instance_data = []
tex_quad_instance_stride = 13 * ctypes.sizeof(ctypes.c_float)
tex_quad_instance_vbo = ctypes.c_uint()
glGenBuffers(1, ctypes.byref(tex_quad_instance_vbo))
glBindBuffer(GL_ARRAY_BUFFER, tex_quad_instance_vbo)
# Set instanced attributes (locations 2+; divisor=1 for per-instance)
# pos (loc 2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, tex_quad_instance_stride, ctypes.c_void_p(0))
glEnableVertexAttribArray(2)
glVertexAttribDivisor(2, 1)
# size (loc 3)
glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, tex_quad_instance_stride, ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(3)
glVertexAttribDivisor(3, 1)
# uv offset (loc 4)
glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, tex_quad_instance_stride, ctypes.c_void_p(5*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(4)
glVertexAttribDivisor(4, 1)
# uv size (loc 5)
glVertexAttribPointer(5, 2, GL_FLOAT, GL_FALSE, tex_quad_instance_stride, ctypes.c_void_p(7*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(5)
glVertexAttribDivisor(5, 1)
# color (loc 6)
glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, tex_quad_instance_stride, ctypes.c_void_p(9*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(6)
glVertexAttribDivisor(6, 1)

glBindVertexArray(0)
glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

# VAO for quads without texture
quad_VAO = ctypes.c_uint()
glGenVertexArrays(1, ctypes.byref(quad_VAO))
glBindVertexArray(quad_VAO)
glBindBuffer(GL_ARRAY_BUFFER, quad_VBO)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quad_EBO)
# position (loc 0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
glEnableVertexAttribArray(0)
# tex (loc 1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(1)
# instances
quads_changed = False
quad_instance_count = 0
quad_instance_data = [0] * 9 # first data is for cursor, later data for selection and other quads
quad_instance_stride = 9 * ctypes.sizeof(ctypes.c_float)
quad_instance_vbo = ctypes.c_uint()
glGenBuffers(1, ctypes.byref(quad_instance_vbo))
glBindBuffer(GL_ARRAY_BUFFER, quad_instance_vbo)
# Set instanced attributes (locations 2+; divisor=1 for per-instance)
# pos (loc 2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, quad_instance_stride, ctypes.c_void_p(0))
glEnableVertexAttribArray(2)
glVertexAttribDivisor(2, 1)
# size (loc 3)
glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, quad_instance_stride, ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(3)
glVertexAttribDivisor(3, 1)
# color (loc 4)
glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, quad_instance_stride, ctypes.c_void_p(5*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(4)
glVertexAttribDivisor(4, 1)

glBindVertexArray(0)
glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

# fps = None
# frame_count = 0
# last_frame_time = time.time()

# if PRINT_TIMINGS: print(f"{-LASTTIME + (LASTTIME:=time.time()):.3f}: Initialization")

texquadShader = Shader(p:=Path(__file__).parent / "spiritstream/shaders/texquad.vert", p.parent / "texquad.frag", ["glyphAtlas", "scale", "offset"])
texquadShader.setUniform("glyphAtlas", 0, "1i")  # 0 means GL_TEXTURE0)
quadShader = Shader(p:=Path(__file__).parent / "spiritstream/shaders/quad.vert", p.parent / "quad.frag", ["scale", "offset"])

# if PRINT_TIMINGS: print(f"{-LASTTIME + (LASTTIME:=time.time()):.3f}: Loading shaders")

glEnable(GL_DEPTH_TEST)
glClearDepth(1)
glClearColor(0, 0, 0, 1)

with open(TEXTPATH, "r") as f: markdown = parse(t:=f.read())
with open(CSSPATH, "r") as f: css = parseCSS(f.read())
default_css = {"body":{"font-size": (16, "px")}}
for rule, declarations in default_css.items():
    for k,v in declarations.items(): css.setdefault(rule, declarations).setdefault(k, v)
frame = Node(K.FRAME, SCENE, [markdown], text=t)
markdown.parent = frame

# load fonts. TODO: multiple @font-face not supported. also, should probably load dynamically when used :(
if "@font-face" in css:
    name = (font:=css["@font-face"])["font-family"]
    if (src:=font["src"])["format"] != "truetype": raise NotImplementedError
    SS.fonts[name] = Font(Path(CSSPATH).parent.joinpath(src["url"]).resolve())

render_tree(frame, css)
SCENE.children = [frame]
show(SS)


# if PRINT_TIMINGS: print(f"{-LASTTIME + (LASTTIME:=time.time()):.3f}: Parse, render text+css")

while not glfwWindowShouldClose(window):
    if glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS: glfwSetWindowShouldClose(window, GLFW_TRUE)
    
    # FPS
    # frame_count += 1
    # current_time = time.time()
    # if (delta:=current_time - last_frame_time) >= 1.0:
    #     fps = int((frame_count / delta + 0.5) // 1)
    #     frame_count = 0
    #     last_frame_time = current_time
    #     print(f"FPS: {fps}", end="\r")
    # if Scene.resized:
    #     offset_vector = vec2(Scene.x * 2 / Scene.w, Scene.y * 2 / Scene.h) # offset applied to all objects. unit in opengl coordinates
    #     Scene.resized = False
    #     glViewport(0, 0, Scene.w, Scene.h)
    # if Scene.scrolled:
    #     offset_vector = vec2(Scene.x * 2 / Scene.w, Scene.y * 2 / Scene.h) # offset applied to all objects. unit in opengl coordinates
    #     if text.selection.instance_count > 0: text.selection.update()
    #     Scene.scrolled = False
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    offset_vector = vec2(SCENE.x * 2 / SS.w, SCENE.y * 2 / SS.h)
    scale, offset = (2 / SS.w, -2 / SS.h), tuple((vec2(-1, 1) + offset_vector).components())

    # NOTE: quads are rendered before textured quads because glyphs (textured quads) have transparency.

    # quads
    if quads_changed: # upload buffer
        quad_instance_data_ctypes = (ctypes.c_float * len(quad_instance_data))(*quad_instance_data)
        glBindBuffer(GL_ARRAY_BUFFER, quad_instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(quad_instance_data_ctypes), quad_instance_data_ctypes, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        quads_changed = False
    quadShader.use()
    quadShader.setUniform("scale", scale, "2f") # inverted y axis. from my view (0,0) is the top left corner, like in browsers
    quadShader.setUniform("offset", offset, "2f")
    glBindVertexArray(quad_VAO)
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, quad_instance_count)
    
    # texture quads
    glBindTexture(GL_TEXTURE_2D, SS.glyphAtlas.texture)
    if tex_quads_changed: # upload buffer. NOTE: no mechanism shrinks the buffer if fewer quads are needed as before. same applies to quad buffer
        tex_quad_instance_data_ctypes = (ctypes.c_float * len(tex_quad_instance_data))(*tex_quad_instance_data)
        glBindBuffer(GL_ARRAY_BUFFER, tex_quad_instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(tex_quad_instance_data_ctypes), tex_quad_instance_data_ctypes, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        tex_quads_changed = False
    texquadShader.use()
    texquadShader.setUniform("scale", scale, "2f") # inverted y axis. from my view (0,0) is the top left corner, like in browsers
    texquadShader.setUniform("offset", offset, "2f")
    glBindVertexArray(tex_quad_VAO)
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, tex_quad_instance_count)
    
    if (error:=glGetError()) != GL_NO_ERROR: print(f"OpenGL Error: {hex(error)}")

    glfwSwapBuffers(window)
    # if PRINT_TIMINGS and LASTTIME is not None:
    #     print(f"{time.time() - LASTTIME:.3f}: First frame done")
    #     print(f"{time.time() - FIRSTTIME:.3f}: Total\n------")
    #     LASTTIME = None
    glfwWaitEvents()

if SAVE_GLYPHATLAS:
    from spiritstream.image import Image
    Image.write(list(reversed(SS.glyphAtlas.bitmap)), Path(__file__).parent / "GlyphAtlas.bmp")

if CACHE_GLYPHATLAS:
    with open(atlaspath, "wb") as f: pickle.dump(SS.glyphAtlas, f)

glDeleteBuffers(1, quad_VBO)
glDeleteBuffers(1, quad_EBO)
glDeleteVertexArrays(1, tex_quad_VAO)
glDeleteVertexArrays(1, quad_VAO)
glDeleteBuffers(1, tex_quad_instance_vbo)
glDeleteBuffers(1, quad_instance_vbo)
texquadShader.delete()
quadShader.delete()
glfwTerminate()