import math, time
from spiritstream.vec import vec2
from spiritstream.bindings.glfw import *
from spiritstream.bindings.opengl import *
from spiritstream.font import Font, Glyph
from typing import Union, List, Dict
from dataclasses import dataclass
from spiritstream.shader import Shader
from spiritstream.textureatlas import TextureAtlas

class Cursor:
    def __init__(self, text:"Text"):
        self.text = text # keep reference for cursor_coords and linewraps
        self.left = None
        self.pos = vec2()
        self.idx = None
        self.line = None
        self.x = 0 # Moving up and down using arrow keys can lead to drifting left or right. To avoid, stores original x in this variable.
    
        self.VBO, self.VAO, self.EBO = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        glGenBuffers(1, ctypes.byref(self.VBO))
        glGenVertexArrays(1, ctypes.byref(self.VAO))
        glGenBuffers(1, ctypes.byref(self.EBO))
        
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(1)

        glBufferData(GL_ARRAY_BUFFER, 80, None, GL_DYNAMIC_DRAW) # pre allocate buffer

        self.indices = [0, 1, 2, 1, 2, 3]
        cursor_indices_ctypes = (ctypes.c_uint * len(self.indices)) (*self.indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(cursor_indices_ctypes), cursor_indices_ctypes, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        self.update(0)

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
        self.pos = vec2(-2, cursorcoords[(self.idx + 1) % len(cursorcoords)].y) if left else cursorcoords[self.idx]
        
        # scroll into view
        if self.pos.y - self.text.lineheightUsed < Scene.y:
            Scene.y = self.pos.y - self.text.lineheightUsed
            Scene.scrolled = True
        elif self.pos.y > Scene.y + Scene.h:
            Scene.y = self.pos.y - Scene.h
            Scene.scrolled = True

        g = Scene.fonts[self.text.font].glyph("|", self.text.fontsize, Scene.dpi)
        k = "|" + str(self.text.fontsize)
        u, v, w, h = Scene.glyphAtlas.coordinates[k] if k in Scene.glyphAtlas.coordinates else Scene.glyphAtlas.add(k, Scene.fonts[self.text.font].render("|", self.text.fontsize, Scene.dpi))
        vertices = [
            # x,y                                            z    u    v
            *(self.pos - vec2(0, g.size.y)).components(),    0.2, u,   v+h, # top left
            *(self.pos + g.size * vec2(1, -1)).components(), 0.2, u+w, v+h, # top right
            *self.pos.components(),                          0.2, u,   v,   # bottom left
            *(self.pos + vec2(g.size.x, 0)).components(),    0.2, u+w, v,   # bottom right
        ]
        if allow_drift: self.x = self.pos.x

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        vertices_ctypes = (ctypes.c_float * len(vertices))(*vertices)
        assert ctypes.sizeof(vertices_ctypes) == 80, "This size is this hard coded in the preallocation"
        glBufferSubData(GL_ARRAY_BUFFER, 0, 80, vertices_ctypes)
        glBindBuffer(GL_ARRAY_BUFFER, 0)


class Selection:
    def __init__(self, text:"Text"):
        self.text = text
        self.start = self.startleft = self.end = self.endleft = self.pos1 = self.pos1left = None
        self.quads = 100 # preallocates for this many. reallocates if necessary
        self.length = 0 # number of quads to actually draw
        
        self.VBO, self.VAO, self.EBO = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        glGenBuffers(1, ctypes.byref(self.VBO))
        glGenVertexArrays(1, ctypes.byref(self.VAO))
        glGenBuffers(1, ctypes.byref(self.EBO))
        
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)

        glBufferData(GL_ARRAY_BUFFER, self.quads * 48, None, GL_DYNAMIC_DRAW) # pre allocate buffer
        self.indices = [x for i in range(self.quads) for x in [i*4, i*4+1, i*4+2, i*4+1, i*4+2, i*4+3]]
        indices_ctypes = (ctypes.c_uint * len(self.indices)) (*self.indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(indices_ctypes), indices_ctypes, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
    
    def update(self, pos1=None, left1=None, pos2=None, left2=None):
        if pos1 == left1 == pos2 == left2 == None: # all are set or none are set. this allows calling selection.update() to rerender it
            assert all([i != None for i in [self.pos1, self.pos1left, self.start, self.startleft, self.end, self.endleft]])
            pos1, left1, pos2, left2 = self.start, self.startleft, self.end, self.endleft
        else:
            assert all([i != None for i in [pos1, left1, pos2, left2]])
            if self.pos1 == None: self.pos1, self.pos1left = pos1, left1
            self.start, self.startleft, self.end, self.endleft = (self.pos1, left1, pos2, left2) if self.pos1 < pos2 else (pos2, left2, self.pos1, left1)
            if self.start == self.end: return self.reset()
        vertices = []
        for i, l in enumerate((l for l in self.text.visible_lines if l.end >= self.start)):
            if i == 0 and l.start <= self.start:
                if self.startleft and l.end == self.start: continue
                vertices.extend([
                    *(self.text.cursorcoords[self.start]).components(),                                   0,
                    *(self.text.cursorcoords[self.start] - vec2(0, self.text.fontsize)).components(), 0,
                ])
            else:
                vertices.extend([
                    *(vec2(0, l.y)).components(),                          0,
                    *(vec2(0, l.y - self.text.fontsize)).components(), 0
                ])
            if self.end <= l.end:
                vertices.extend([
                    *(self.text.cursorcoords[self.end]).components(),                                   0,
                    *(self.text.cursorcoords[self.end] - vec2(0, self.text.fontsize)).components(), 0,
                ])
                break
            else:
                vertices.extend([
                    *(self.text.cursorcoords[l.end] + vec2(5, 0)).components(),                       0,
                    *(self.text.cursorcoords[l.end] - vec2(-5, self.text.fontsize)).components(), 0,
                ])
        assert len(vertices) < self.quads*12, "buffer extension not yet supported"
        assert len(vertices) % 12 == 0
        self.length = len(vertices) // 12 # number of quads to render

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        vertices_ctypes = (ctypes.c_float * len(vertices))(*vertices)
        glBufferSubData(GL_ARRAY_BUFFER, 0, ctypes.sizeof(vertices_ctypes), vertices_ctypes)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def reset(self):
        self.length = 0
        self.start = self.end = self.pos1 = self.startleft = self.pos1left = self.endleft = None

@dataclass
class Line:
    newline:bool # if True, line was preceeded by "\n", else its a continuation of the previous one
    y:float
    start:int # cursor idx where line starts
    end:int

class Text:
    def __init__(self, text:str, x, y, w, h):
        self.text = text
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.font = Scene.font
        self.fontsize = Scene.fontsize
        self.lineheight = Scene.lineheight
        self.lineheightUsed = self.fontsize * self.lineheight

        self.lines = [] # useful for determining which portion of the text is currently visible, cursor movement and editing relating to lines
        self.cursorcoords = []

        self.VBO, self.VAO, self.EBO = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        glGenBuffers(1, ctypes.byref(self.VBO))
        glGenVertexArrays(1, ctypes.byref(self.VAO))
        glGenBuffers(1, ctypes.byref(self.EBO))

        self.update()
        self.cursor = Cursor(self)

        self.selection = Selection(self)
        Scene.nodes.append(self) # is this a good idea?

    def goto(self, pos:Union[int, vec2], allow_drift=True, left=False, selection=False):
        oldidx, oldleft  = self.cursor.idx, self.cursor.left
        self.cursor.update(pos, allow_drift=allow_drift, left=left)
        if selection: self.selection.update(oldidx, oldleft, self.cursor.idx, self.cursor.left)
        else: self.selection.reset()

    @property
    def visible_lines(self): return (l for l in self.lines if Scene.y + Scene.h + self.lineheightUsed >= l.y > self.y - self.lineheightUsed)

    def update(self):
        self.lines = []
        vertices = []
        indices = []
        offset = vec2(0, self.lineheightUsed) # bottom left corner of character in the first line
        cursorcoords = []
        newline = True
        for i, char in enumerate(self.text):
            cursorcoords.append((offset-vec2(2, 0)).copy()) # move cursor 2 pixel to left of char
            if ord(char) == 10:  # newline
                self.lines.append(Line(newline, offset.y, 0 if len(self.lines) == 0 else self.lines[-1].end + (1 if newline else 0), i))
                newline = True
                offset = vec2(0, offset.y + self.lineheightUsed)
                continue
            g = Scene.fonts[self.font].glyph(char, self.fontsize, Scene.dpi)
            if offset.x + g.advance > self.w:
                assert i > 0
                self.lines.append(Line(newline, offset.y, 0 if len(self.lines) == 0 else self.lines[-1].end + (1 if newline else 0), i))
                newline = False
                offset = vec2(0, offset.y + self.lineheightUsed) # new line, no word splitting
            if ord(char) == 32: # space
                offset.x += g.advance
                continue
            k = char + str(self.fontsize)
            u, v, w, h = Scene.glyphAtlas.coordinates[k] if k in Scene.glyphAtlas.coordinates else Scene.glyphAtlas.add(k, Scene.fonts[self.font].render(char, self.fontsize, Scene.dpi))
            vertices += [
                # x                                              y                                                z  u,   v
                (top_left_x := self.x + offset.x + g.bearing.x), (top_left_y := self.y + offset.y - g.bearing.y), 0, u,   v+h, # top left
                top_left_x + g.size.x,                           top_left_y,                                      0, u+w, v+h, # top right
                top_left_x,                                      top_left_y + g.size.y,                           0, u,   v, # bottom left
                top_left_x + g.size.x,                           top_left_y + g.size.y,                           0, u+w, v,  # bottom right
            ]
            last = len(vertices)//5 - 1 # 5 because xyz and texture xy makes 5 values per vertex
            indices += [
                last-3, last-2, last-1, # triangle top left - top right - bottom left
                last-2, last-1, last    # triangle top right - bottom left - bottom right
            ]
            offset.x += g.advance
        self.lines.append(Line(newline, offset.y, self.lines[-1].end + (1 if newline else 0) if len(self.lines) > 0 else 0, len(cursorcoords)))
        cursorcoords.append((offset-vec2(2, 0)).copy()) # first position after the last character
        self.cursorcoords, self.indices = cursorcoords, indices

        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)

        vertices_ctypes = (ctypes.c_float * len(vertices))(*vertices)
        glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(vertices_ctypes), vertices_ctypes, GL_DYNAMIC_DRAW)

        indices_ctypes = (ctypes.c_uint * len(indices)) (*indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(indices_ctypes), indices_ctypes, GL_DYNAMIC_DRAW)
        
        # vertices
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # texture coordinate
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def write(self, t:Union[int, str]):
        if t == None: return # can happen on empty clipboard
        if isinstance(t, int): t = chr(t) # codepoint
        if self.selection.length == 0:
            self.text = self.text[:self.cursor.idx] + t + self.text[self.cursor.idx:]
            self.update()
            self.cursor.update(self.cursor.idx + len(t))
        else:
            self.text = self.text[:self.selection.start] + t + self.text[self.selection.end:]
            self.update()
            self.cursor.update(self.selection.start + len(t))
            self.selection.reset()

    def erase(self, right=False):
        if self.selection.length == 0:
            offset = 1 if right else 0
            self.text = self.text[:self.cursor.idx-1 + offset] + self.text[self.cursor.idx + offset:]
            self.update()
            self.cursor.update(self.cursor.idx - 1 + offset)
        else:
            self.text = self.text[:self.selection.start] + self.text[self.selection.end:]
            self.update()
            self.cursor.update(self.selection.start)
            self.selection.reset()

@GLFWframebuffersizefun
def framebuffer_size_callback(window, width, height):
    Scene.w, Scene.h, Scene.resized = width, height, True

@GLFWcharfun
def char_callback(window, codepoint): text.write(codepoint)

@GLFWkeyfun
def key_callback(window, key:int, scancode:int, action:int, mods:int):
    if action in [GLFW_PRESS, GLFW_REPEAT]:
        selection = bool(mods & GLFW_MOD_SHIFT)
        if key == GLFW_KEY_LEFT: text.goto(text.selection.start) if text.selection.start != None and not selection else text.goto(text.cursor.idx - 1, left=text.cursor.idx - 1 == text.cursor.line.start and not text.cursor.line.newline, selection=selection)
        if key == GLFW_KEY_RIGHT: text.goto(text.selection.end) if text.selection.end != None and not selection else text.goto(text.cursor.idx+1, selection=selection)
        if key == GLFW_KEY_UP: text.goto(text.cursor.pos - vec2(0, text.lineheightUsed), allow_drift=False, selection=selection)
        if key == GLFW_KEY_DOWN: text.goto(text.cursor.pos + vec2(0, text.lineheightUsed), allow_drift=False, selection=selection)
        if key == GLFW_KEY_BACKSPACE: text.erase()
        if key == GLFW_KEY_DELETE: text.erase(right=True)
        if key == GLFW_KEY_ENTER: text.write(ord("\n"))
        if key == GLFW_KEY_S:
            if mods & GLFW_MOD_CONTROL: # SAVE
                with open("text.txt", "w") as f: f.write(text.text)
        if key == GLFW_KEY_A and mods & GLFW_MOD_CONTROL:  # select all
            text.selection.reset()
            text.goto(0)
            text.goto(len(text.text), selection=True)
        if key == GLFW_KEY_C and mods & GLFW_MOD_CONTROL and text.selection.length > 0: glfwSetClipboardString(window, text.text[text.selection.start:text.selection.end].encode()) # copy
        if key == GLFW_KEY_V and mods & GLFW_MOD_CONTROL: text.write(glfwGetClipboardString(window).decode()) # paste
        if key == GLFW_KEY_X and mods & GLFW_MOD_CONTROL and text.selection.length > 0: # cut
            glfwSetClipboardString(window, text.text[text.selection.start:text.selection.end].encode()) # copy
            text.erase()
        if key == GLFW_KEY_HOME: text.goto(text.cursor.line.start, allow_drift=True, left=True, selection=selection)
        if key == GLFW_KEY_END: text.goto(text.cursor.line.end, allow_drift=True, selection=selection) # TODO: double pressing home in wrapping lines, works, but not double pressing end

@GLFWmousebuttonfun
def mouse_callback(window, button:int, action:int, mods:int):
    if button == GLFW_MOUSE_BUTTON_1 and action in [GLFW_PRESS, GLFW_RELEASE]:
        x, y = ctypes.c_double(), ctypes.c_double()
        glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
        text.goto(vec2(x.value - text.x, y.value + text.y), selection=action == GLFW_RELEASE or glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) # adjust for offset vector

@GLFWcursorposfun
def cursor_pos_callback(window, x, y):
    if glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS:
        x, y = ctypes.c_double(), ctypes.c_double()
        glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
        text.goto(vec2(x.value - text.x, y.value + text.y), selection=True) # adjust for offset vector

@GLFWscrollfun
def scroll_callback(window, x, y):
    Scene.y -= y * 40
    Scene.scrolled = True

class _Scene:
    def __init__(self):
        self.title = "Spiritstream"
        self.x = 0 # horizontal scrolling not implemented
        self.y = 0
        self.w = 700
        self.h = 1000
        self.dpi = 96
        self.resized = True
        self.scrolled = False
        self.font = None
        self.fonts = dict()
        self.fontsize = 12
        self.lineheight = 1.5
        self.glyphAtlas:TextureAtlas # GL_RED because single channel
        self.nodes = []
Scene = _Scene()

glfwInit()
glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, True)
window = glfwCreateWindow(Scene.w, Scene.h, Scene.title.encode("utf-8"), None, None)
glfwMakeContextCurrent(window)
Scene.glyphAtlas = TextureAtlas(GL_RED) # assign after window creation + assignment else texture settings won't apply

glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)
glfwSetCharCallback(window, char_callback)
glfwSetKeyCallback(window, key_callback)
glfwSetMouseButtonCallback(window, mouse_callback)
glfwSetCursorPosCallback(window, cursor_pos_callback)
glfwSetScrollCallback(window, scroll_callback)

Scene.fonts["Fira Code"] = Font("assets/fonts/Fira_Code_v6.2/ttf/FiraCode-Regular.ttf")
Scene.font = "Fira Code"

text_width = 500
with open("text.txt", "r") as f: text = Text(f.read(), (Scene.w-text_width)/2, 0, text_width, Scene.h)

from ascii_glyphatlas import write_bmp
write_bmp("glyph atlas.bmp", list(reversed(Scene.glyphAtlas.bitmap)))

vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform vec2 scale;
uniform vec2 offset;

out vec2 texCoord;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0) * vec4(scale, 1, 1) + vec4(offset, 0, 0);
    texCoord = vec2(aTexCoord.x, aTexCoord.y);
}"""

fragment_shader_source = """
#version 330 core
in vec2 texCoord;
out vec4 FragColor;

uniform sampler2D glyphTexture;
uniform vec3 textColor;

void main()
{
    FragColor = vec4(textColor, texture(glyphTexture, texCoord).r);
}"""

textShader = Shader(vertex_shader_source, fragment_shader_source, ["glyphTexture", "textColor", "scale", "offset"])
textShader.setUniform("glyphTexture", 0, "1i")  # 0 means GL_TEXTURE0)

vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform vec2 scale;
uniform vec2 offset;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0) * vec4(scale, 1, 1) + vec4(offset, 0, 0);
}"""

fragment_shader_source = """
#version 330 core
out vec4 FragColor;

uniform vec3 selectionColor;

void main()
{
    FragColor = vec4(selectionColor, 1.0);
}"""
selectionShader = Shader(vertex_shader_source, fragment_shader_source, ["selectionColor", "offset", "scale"])
selectionShader.setUniform("selectionColor", (0x42 / 0xff, 0x30 / 0xff, 0x24 / 0xff), "3f")

fps = None
frame_count = 0
last_frame_time = time.time()

glClearColor(0, 0, 0, 1)
glBindTexture(GL_TEXTURE_2D, Scene.glyphAtlas.texture)

while not glfwWindowShouldClose(window):
    if glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS: glfwSetWindowShouldClose(window, GLFW_TRUE)
    
    # FPS
    frame_count += 1
    current_time = time.time()
    if (delta:=current_time - last_frame_time) >= 1.0:
        fps = int((frame_count / delta + 0.5) // 1)
        frame_count = 0
        last_frame_time = current_time
        print(f"FPS: {fps}", end="\r")
    if Scene.resized:
        text.x = (Scene.w - text.w) / 2
        offset_vector = vec2(Scene.x * 2 / Scene.w, Scene.y * 2 / Scene.h) # offset applied to all objects. unit in opengl coordinates
        Scene.resized = False
        glViewport(0, 0, Scene.w, Scene.h)
    if Scene.scrolled:
        offset_vector = vec2(Scene.x * 2 / Scene.w, Scene.y * 2 / Scene.h) # offset applied to all objects. unit in opengl coordinates
        if text.selection.length > 0: text.selection.update()
        Scene.scrolled = False
    
    glClear(GL_COLOR_BUFFER_BIT)
    
    selectionShader.use()
    selectionShader.setUniform("scale", (2 / Scene.w, -2 / Scene.h), "2f") # inverted y axis. from my view (0,0) is the top left corner, like in browsers
    selectionShader.setUniform("offset", tuple((vec2(-1, 1) + offset_vector).components()), "2f")
    glBindVertexArray(text.selection.VAO)
    glDrawElements(GL_TRIANGLES, text.selection.length * 6, GL_UNSIGNED_INT, 0)

    textShader.use()
    textShader.setUniform("scale", (2 / Scene.w, -2 / Scene.h), "2f") # inverted y axis. from my view (0,0) is the top left corner, like in browsers
    textShader.setUniform("offset", tuple((vec2(-1, 1) + offset_vector).components()), "2f")
    textShader.setUniform("textColor", (1.0, 0.5, 0.2), "3f")
    glBindVertexArray(text.VAO)
    glDrawElements(GL_TRIANGLES, len(text.indices), GL_UNSIGNED_INT, 0)

    textShader.setUniform("textColor", (1.0, 1.0, 1.0), "3f")
    glBindVertexArray(text.cursor.VAO)
    glDrawElements(GL_TRIANGLES, len(text.cursor.indices), GL_UNSIGNED_INT, 0)


    if (error:=glGetError()) != GL_NO_ERROR: print(f"OpenGL Error: {hex(error)}")

    glfwSwapBuffers(window)
    glfwWaitEvents()

glDeleteVertexArrays(1, text.VAO)
glDeleteBuffers(1, text.VBO)
glDeleteBuffers(1, text.EBO)
glDeleteVertexArrays(1, text.cursor.VAO)
glDeleteBuffers(1, text.cursor.VBO)
glDeleteBuffers(1, text.cursor.EBO)
textShader.delete()
glfwTerminate()