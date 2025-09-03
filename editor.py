from pprint import pprint
import math, time, pickle, functools
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
from spiritstream.tree import parse, parseCSS, Node, K, Color, color_from_hex, walk, show, INLINE_NODES, steal_children

if PRINT_TIMINGS: FIRSTTIME = LASTTIME = time.time()

TEXTPATH = "text.txt"
CSSPATH = "test/test.css"
FORMATTING_COLOR = color_from_hex(0xaaa)
CURSOR_COLOR = color_from_hex(0xfff)

"""
selection quads per node (could try and merge)

the deepest node where the cursor is, gets edit=True. keep set of those nodes to quickly turn edit off again.
also applies nodes under selection area
if cursor position is between two nodes where the xs are also shared (necessary to avoid line-breaking appearance of contiguousness), both get edit=True.
If cursor._find hits a node, which can happen if walking horizontally into a new line. then set edit=True for that node, rerender and 
upon rerendering remove lines whos parent is not TEXT

Node
    kind:Kinds
    parent:Node
    children:List[Node]
    data:Dict[]

text edit trigger rerender. edit only happens in raw text or piece table. text stored in frame node.
"""

class Cursor:
    def __init__(self, frame:Node):
        self.frame = frame # keep reference for cursor_coords and linewraps
        self.idx = 0
        self.pos:vec2
        self.x = None
        self.node:Node
        
        self.update(self.idx)

    def _find(self, node:Node, pos:Union[int, vec2], up:bool=False) -> Node:
        if hasattr(node, "children"):
            if isinstance(pos, int):
                for c in node.children:
                    if c.start <= pos and ((pos <= c.end and up) or (not up and pos < c.end)): return self._find(c, pos, up)
                return node if up else self._find(node, pos, True) # try up before giving up
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

    def update(self, pos:Union[int, vec2], allow_drift:bool=True, up=False):
        """
        if pos is vec2, get the int. this might fail because _find returns node.
        In that case, it adds edit=True to the node (and neighbors) and triggers populate_render_data and then updates itself again with pos:vec2
        NOTE: using up=True only works if the line is wrapped. else node.xs[pos-node.start] will be out of range
        """
        global quads_changed
        if isinstance(pos, int): pos = max(min(pos, len(self.frame.text)), 0)
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
                self.update(pos, allow_drift=allow_drift)
            else: raise RuntimeError(f"Could not resolve position {pos} to LINE after enabling edit mode on {node}")
            return
        
        self.node = node
        new_editnodes = set()
        rerender = False
        p1 = _find_edit_parent(node, self.idx)
        if p1.parent.k not in [K.P, K.EMPTY_LINE]: # edit mode changes nothing here, saves some rerenders
            if not _in_edit_view(p1): rerender = p1.edit = True
            if p1.edit: new_editnodes.add(p1)
        # if cursor between nodes on same line, edit=True on both
        node2 = self._find(self.frame.children[0], self.idx, up=self.idx == node.start)
        p2 = _find_edit_parent(node2, self.idx)
        if node.y == node2.y and p2.parent.k not in [K.P, K.EMPTY_LINE]:
            if not _in_edit_view(p2): rerender = p2.edit = True
            if p2.edit: new_editnodes.add(p2)

        for n in self.frame.editnodes:
            if n not in new_editnodes: n.edit = False # If frame in editnodes (everthing rendered in edit view), rerender is never True
        if new_editnodes != self.frame.editnodes: rerender = True
        self.frame.editnodes = new_editnodes
        if rerender:
            populate_render_data(self.frame, SS.css, reset=True)
            self.update(self.idx, allow_drift=allow_drift)
        else:
            if allow_drift: self.x = self.pos.x
            quad_instance_data[:9] = [*self.pos.components(), 0.4, 2, node.h, (c:=CURSOR_COLOR).r, c.g, c.b, c.a]
            quads_changed = True

    def move(self, d:str):
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
            if closest is None: self.move("start")
            else: self.update(closest.start + closest.xs.index(min(closest.xs, key=lambda x: abs(x-pos.x))), allow_drift=False)
        elif d == "right":
            if self.idx in self.frame.wraps and self.idx==self.node.end: self.update(self.idx)
            elif self.idx+1 in self.frame.wraps: self.update(self.idx+1, up=True)
            else: self.update(self.idx + 1)
        elif d == "down":
            pos = vec2(self.pos.x if self.x is None else self.x, self.pos.y)
            lines = (n for n in walk(self.frame) if n.k is K.LINE and n.y > self.node.y)
            closest, dx, dy = None, math.inf, math.inf
            for l in lines:
                dxc = 0 if l.x <= pos.x <= l.x+l.w else min(abs(pos.x-l.x), abs(pos.x-(l.x+l.w)))
                dyc = 0 if l.y <= pos.y <= l.y+l.h else min(abs(pos.y-l.y), abs(pos.y-(l.y+l.h)))
                if dyc < dy: closest, dx, dy = l, dxc, dyc
                elif dyc == dy and dxc < dx: closest, dx = l, dxc
            if closest is None: self.move("end")
            else: self.update(closest.start + closest.xs.index(min(closest.xs, key=lambda x: abs(x-pos.x))), allow_drift=False)
        elif d == "left":
            if self.idx in self.frame.wraps and not self.idx==self.node.end: self.update(self.idx, up=True)
            else: self.update(self.idx - 1)
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
            if idx is not None: self.update(idx)
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
                if idx in self.frame.wraps or idx == len(self.frame.text): self.update(idx, up=True)
                else: self.update(idx - 1)
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
    else:
        assert node.children, node
        if node.start <= idx < node.children[0].start: return next((n for n in node.children if n.k is K.TEXT))
        elif node.children[-1].end < idx <= node.end: return next((n for n in reversed(node.children) if n.k is K.TEXT))
    raise NotImplementedError

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

def typeset(text:str, x:float, y:float, width:float, font:str, fontsize:float, color:Color, lineheight:float, newline_x:float=None, align="left",
            start=0) -> Tuple[Node, List[float]]:
    global tex_quad_instance_data, tex_quad_instance_count, tex_quad_instance_stride
    lines, idata, wraps = _typeset(text, x, y, width, font, fontsize, color, lineheight, newline_x, align, start)
    tex_quad_instance_data[(idx:=tex_quad_instance_count*tex_quad_instance_stride):idx+len(idata)] = idata
    tex_quad_instance_count += int(len(idata) / tex_quad_instance_stride)
    frame.wraps.extend(wraps)
    return lines

# TODO: select font more carefully. may contain symbols not in font
# something like g = next((fonŧ[g] for font in fonts if g in font))
# TODO: line_strings is creating ambiguity, sucks, remove. index into text using line start and end to actually get the text. obviously.
@functools.cache # TODO custom cache so it can also shift cached lines down
def _typeset(text:str, x:float, y:float, width:float, font:str, fontsize:float, color:Color, lineheight:float, newline_x:float=None, align="left",
            start=0) -> Tuple[Node, List[float]]:
    """
    Returns
    - Node with LINE children to add to the node tree
    - Tex quad instance data for all glyphs
    """
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
        g = SS.fonts[font].glyph(c, fontsize, 72) # NOTE: dpi is 72 so the font renderer does no further scaling. It uses 72 dpi baseline.
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
    ascentpx = SS.fonts[font].engine.hhea.ascent * fontsize / SS.fonts[font].engine.head.unitsPerEM
    descentpx = SS.fonts[font].engine.hhea.descent * fontsize / SS.fonts[font].engine.head.unitsPerEM
    total = ascentpx - descentpx
    for s, l in zip(line_strings, lines.children):
        for i, c in enumerate(s):
            g = SS.fonts[font].glyph(c, fontsize, 72)
            if c != " ":
                key = f"{font}_{ord(c)}_{fontsize}"
                instance_data += [l.xs[i] + g.bearing.x, l.y + ascentpx + (lineheight - total)/2 + (g.size.y - g.bearing.y), 0.5, # pos
                    g.size.x, -g.size.y, # size
                    *(SS.glyphAtlas.coordinates[key] if key in SS.glyphAtlas.coordinates else SS.glyphAtlas.add(key, SS.fonts[font].render(c, fontsize, 72))), # uv offset and size
                    color.r, color.g, color.b, color.a] # color 
    return lines, instance_data, wraps


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

@GLFWcharfun
def char_callback(window, codepoint): write(frame, chr(codepoint))

def erase(frame:Node, right=False):
    frame.text = frame.text[:max(frame.cursor.idx - (0 if right else 1), 0)] + frame.text[(frame.cursor.idx + (1 if right else 0)):]
    markdown = parse(frame.text)
    frame.children = [markdown]
    markdown.parent = frame
    populate_render_data(frame, SS.css, reset=True)
    frame.cursor.update(frame.cursor.idx - (0 if right else 1))

def write(frame:Node, text:str):
    t0 = time.time()
    frame.text = frame.text[:frame.cursor.idx] + text + frame.text[frame.cursor.idx:]
    markdown = parse(frame.text)
    frame.children = [markdown]
    markdown.parent = frame
    populate_render_data(frame, SS.css, reset=True)
    frame.cursor.update(frame.cursor.idx + len(text))
    print(f"{time.time() - t0:.3f}")

@GLFWkeyfun
def key_callback(window, key:int, scancode:int, action:int, mods:int):
    if action in [GLFW_PRESS, GLFW_REPEAT]:
        selection = bool(mods & GLFW_MOD_SHIFT)
        if key == GLFW_KEY_LEFT: frame.cursor.move("left")
        elif key == GLFW_KEY_RIGHT: frame.cursor.move("right")
        elif key == GLFW_KEY_UP: frame.cursor.move("up")
        elif key == GLFW_KEY_DOWN: frame.cursor.move("down")
        if key == GLFW_KEY_BACKSPACE: erase(frame)
        if key == GLFW_KEY_DELETE: erase(frame, right=True)
        if key == GLFW_KEY_ENTER: write(frame, "\n")
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
        elif key == GLFW_KEY_HOME: frame.cursor.move("start")
        elif key == GLFW_KEY_END: frame.cursor.move("end")

@GLFWmousebuttonfun
def mouse_callback(window, button:int, action:int, mods:int):
    if button == GLFW_MOUSE_BUTTON_1 and action is GLFW_PRESS:
        x, y = ctypes.c_double(), ctypes.c_double()
        glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
        frame.cursor.update(vec2(x.value - SCENE.x, y.value - SCENE.y))
        # text.goto(vec2(x.value - text.x, y.value + text.y), selection=action == GLFW_RELEASE or glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) # adjust for offset vector

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
    global quad_instance_count, tex_quad_instance_count, quads_changed, tex_quads_changed, quad_instance_stride, tex_quad_instance_stride
    if reset:
        quads_changed = tex_quads_changed = True
        quad_instance_count = 1 
        tex_quad_instance_count = 0
        to_delete = list(filter(lambda x: x.k is K.LINE, walk(node)))
        for l in to_delete: l.parent.children.remove(l)
        frame.wraps = []
    if node.k is K.FRAME: text = node.text
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
            if node.parent.k is K.LI and not edit_view:
                if node.parent.parent.k is K.UL:
                    y = node.y + style["font-size"] * 0.066666 # hack to approximate browser rendering. Shift without changing node.y for talled node.
                    typeset("•", node.x - 1.25 * style["font-size"], y, style["font-size"], style["font-family"], style["font-size"]*1.4,
                            style["color"], style["line-height"]*1.075, align="right", start=node.start)
                elif node.parent.parent.k is K.OL:
                    typeset(d:=f"{node.parent.digit}.", node.x-(len(d)+0.5)*style["font-size"], y, style["font-size"]*len(d), style["font-family"],
                            style["font-size"]*1.05, style["color"], style["line-height"]*1.075, align="right", start=node.start)
            t = text[node.start:node.end].upper() if style.get("text-transform") == "uppercase" else text[node.start:node.end]
            lines = typeset(t, node.x, y, style["_width"], style["font-family"], style["font-size"], style["color"], style["line-height"],
                            pstyle["_block"].x, start=node.start)
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
                                    style["font-size"], FORMATTING_COLOR, style["line-height"], for_children["_block"].x, start=start)
                    edit_view_lines.append((i, lines.children))
                    for_children["_inline"] = vec2((c:=lines.children[-1]).x + c.w, c.y)
                    for_children["_block"] = vec2(style["_block"].x, c.y + c.h)
                
            for_children["_block"], for_children["_inline"], for_children["_margin"] = populate_render_data(child, css_rules, for_children, text)

            if child.k is K.LINE: print(child)
            if i == len(node.children) - 1 and child_edit_view and child.end != node.end:
                lines = typeset(text[child.end:node.end], for_children["_inline"].x, for_children["_inline"].y, style["_width"], style["font-family"],
                                style["font-size"], FORMATTING_COLOR, style["line-height"], for_children["_block"].x, start=child.end)
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
                quad_instance_data[(count:=quad_instance_count)*(stride:=quad_instance_stride):(count+1)*stride] = [node.x, node.y, 0.6, node.w, node.h, (c:=style["background-color"]).r, c.g, c.b, c.a]
                quad_instance_count += 1
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
        elif u == "%": return (pstyle["font-size"] if k == "font-size" else pstyle["_width"]) * v / 100
    return v[0] if isinstance(v, tuple) and len(v) == 1 else v

SS = Node(K.SS, None, resized=True, scrolled=False, fonts={}, glyphAtlas=None, dpi=96, title="Spiritstream", w=700, h=1800)
SCENE = Node(K.SCENE, SS, x=100, y=0)
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
glfwSetCharCallback(window, char_callback)
glfwSetKeyCallback(window, key_callback)
glfwSetMouseButtonCallback(window, mouse_callback)
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
tex_quad_instance_stride = 13
tex_quad_instance_stride_c = 13 * ctypes.sizeof(ctypes.c_float)
tex_quad_instance_vbo = ctypes.c_uint()
glGenBuffers(1, ctypes.byref(tex_quad_instance_vbo))
glBindBuffer(GL_ARRAY_BUFFER, tex_quad_instance_vbo)
# Set instanced attributes (locations 2+; divisor=1 for per-instance)
# pos (loc 2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, tex_quad_instance_stride_c, ctypes.c_void_p(0))
glEnableVertexAttribArray(2)
glVertexAttribDivisor(2, 1)
# size (loc 3)
glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, tex_quad_instance_stride_c, ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(3)
glVertexAttribDivisor(3, 1)
# uv offset (loc 4)
glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, tex_quad_instance_stride_c, ctypes.c_void_p(5*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(4)
glVertexAttribDivisor(4, 1)
# uv size (loc 5)
glVertexAttribPointer(5, 2, GL_FLOAT, GL_FALSE, tex_quad_instance_stride_c, ctypes.c_void_p(7*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(5)
glVertexAttribDivisor(5, 1)
# color (loc 6)
glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, tex_quad_instance_stride_c, ctypes.c_void_p(9*ctypes.sizeof(ctypes.c_float)))
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
quad_instance_stride = 9
quad_instance_stride_c = 9 * ctypes.sizeof(ctypes.c_float)
quad_instance_vbo = ctypes.c_uint()
glGenBuffers(1, ctypes.byref(quad_instance_vbo))
glBindBuffer(GL_ARRAY_BUFFER, quad_instance_vbo)
# Set instanced attributes (locations 2+; divisor=1 for per-instance)
# pos (loc 2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, quad_instance_stride_c, ctypes.c_void_p(0))
glEnableVertexAttribArray(2)
glVertexAttribDivisor(2, 1)
# size (loc 3)
glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, quad_instance_stride_c, ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(3)
glVertexAttribDivisor(3, 1)
# color (loc 4)
glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, quad_instance_stride_c, ctypes.c_void_p(5*ctypes.sizeof(ctypes.c_float)))
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
with open(CSSPATH, "r") as f: SS.css = parseCSS(f.read())
default_css = {"body":{"font-size": (16, "px")}}
for rule, declarations in default_css.items():
    for k,v in declarations.items(): SS.css.setdefault(rule, declarations).setdefault(k, v)
frame = Node(K.FRAME, SCENE, [markdown], text=t, editnodes=set(), wraps=[], edit=False)
markdown.parent = frame

# load fonts. TODO: multiple @font-face not supported. also, should probably load dynamically when used :(
if "@font-face" in SS.css:
    name = (font:=SS.css["@font-face"])["font-family"]
    if (src:=font["src"])["format"] != "truetype": raise NotImplementedError
    SS.fonts[name] = Font(Path(CSSPATH).parent.joinpath(src["url"]).resolve())

t0 = time.time()
populate_render_data(frame, SS.css, reset=True)
print(f"{time.time() - t0:.3f}")

from spiritstream.tree import serialize
with open("./out.html", "w") as f: f.write(serialize(frame))

frame.cursor = Cursor(frame) # after rendertree so it can find line elements in the tree
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
        quad_instance_data_ctypes = (ctypes.c_float * (length := quad_instance_count * quad_instance_stride))(*quad_instance_data[:length])
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
        tex_quad_instance_data_ctypes = (ctypes.c_float * (length := tex_quad_instance_count * tex_quad_instance_stride))(*tex_quad_instance_data[:length])
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