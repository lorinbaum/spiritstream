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
        self.rpos = vec2() # relative to self.text
        self._apos = vec2() # absolute. derived from self.text.x, self.text.y and rpos. don't set directly or derive from self.rpos.
        self.idx = None
        self.line = None
        self.x = 0 # Moving up and down using arrow keys can lead to drifting left or right. To avoid, stores original x in this variable.

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

class Selection:
    def __init__(self, text:"Text"):
        self.color = color_from_hex(0x423024)
        self.text = text
        self.start = self.startleft = self.end = self.endleft = self.pos1 = self.pos1left = None
        self.instance_count = 0
    
    def update(self, pos1=None, left1=None, pos2=None, left2=None):
        self.instance_count = 0
        done = False # used for existing the for loop below
        if pos1 == left1 == pos2 == left2 == None: # all are set or none are set. this allows calling selection.update() to rerender it
            assert all([i != None for i in [self.pos1, self.pos1left, self.start, self.startleft, self.end, self.endleft]])
            pos1, left1, pos2, left2 = self.start, self.startleft, self.end, self.endleft
        else:
            assert all([i != None for i in [pos1, left1, pos2, left2]])
            if self.pos1 == None: self.pos1, self.pos1left = pos1, left1
            self.start, self.startleft, self.end, self.endleft = (self.pos1, left1, pos2, left2) if self.pos1 < pos2 else (pos2, left2, self.pos1, left1)
            if self.start == self.end: return self.reset()
        for i, l in enumerate((l for l in self.text.visible_lines if l.end >= self.start)):
            if i == 0 and l.start <= self.start:
                if self.startleft and l.end == self.start: continue
                x, y = self.text.cursorcoords[self.start].components()
            else: x, y = 0, l.y
            if self.end <= l.end: w, done = self.text.cursorcoords[self.end].x - x, True 
            else: w = self.text.cursorcoords[l.end].x - x + 5
            instance_data = [x+self.text.x, y+self.text.y+3, 0.9, w, -self.text.lineheightUsed, self.color.r, self.color.g, self.color.b, self.color.a]
            # +1 offset because first instance is cursor
            quad_instance_data[(self.instance_count+1)*(s:=len(instance_data)):(self.instance_count+2)*s] = instance_data
            self.instance_count += 1
            if done: break
        global quads_changed
        quads_changed = True

    def reset(self):
        self.instance_count = 0
        self.start = self.end = self.pos1 = self.startleft = self.pos1left = self.endleft = None

@dataclass
class Line:
    newline:bool # if True, line was preceeded by "\n", else its a continuation of the previous one
    y:float
    start:int # cursor idx where line starts
    end:int

class Text:
    def __init__(self, text:str, x, y, w, h, color):
        self.text = text
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.font = Scene.font
        self.fontsize = Scene.fontsize
        self.lineheight = Scene.lineheight
        self.lineheightUsed = self.fontsize * self.lineheight
        self.lines:List[Line] # useful for determining which portion of the text is currently visible, cursor movement and editing relating to lines
        self.cursorcoords:List[vec2]
        self.instance_count = 0 # number of quad instances. Used for rendering them.

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
        lines = []
        cursorcoords = []
        offset = vec2(0, self.lineheightUsed) # bottom left corner of character in the first line
        newline = True
        instance_count = 0
        for i, char in enumerate(self.text):
            cursorcoords.append(offset.copy())
            if ord(char) == 10:  # newline
                lines.append(Line(newline, offset.y, 0 if len(lines) == 0 else lines[-1].end + (1 if newline else 0), i))
                newline = True
                offset = vec2(0, offset.y + self.lineheightUsed)
                continue
            g = Scene.fonts[self.font].glyph(char, self.fontsize, Scene.dpi)
            if offset.x + g.advance > self.w:
                assert i > 0
                lines.append(Line(newline, offset.y, 0 if len(lines) == 0 else lines[-1].end + (1 if newline else 0), i))
                newline = False
                offset = vec2(0, offset.y + self.lineheightUsed) # new line, no word splitting
            if ord(char) == 32: # space
                offset.x += g.advance
                continue
            k = Scene.font + char + str(self.fontsize)
            u, v, w, h = Scene.glyphAtlas.coordinates[k] if k in Scene.glyphAtlas.coordinates else Scene.glyphAtlas.add(k, Scene.fonts[self.font].render(char, self.fontsize, Scene.dpi))

            instance_data = [
                self.x + offset.x + g.bearing.x, self.y + offset.y + (g.size.y - g.bearing.y), 0.5, # pos
                g.size.x, -g.size.y, # size
                u, v, # uv offset
                w, h, # uv size
                self.color.r, self.color.g, self.color.b, self.color.a # color
            ]
            tex_quad_instance_data[instance_count*(s:=len(instance_data)):(instance_count+1)*s] = instance_data
            global tex_quads_changed
            tex_quads_changed = True
            instance_count += 1
            offset.x += g.advance
        lines.append(Line(newline, offset.y, lines[-1].end + (1 if newline else 0) if len(lines) > 0 else 0, len(cursorcoords)))
        cursorcoords.append((offset-vec2(2, 0)).copy()) # first position after the last character
        self.lines, self.cursorcoords, self.instance_count = lines, cursorcoords, instance_count

    def write(self, t:Union[int, str]):
        if t == None: return # can happen on empty clipboard
        if isinstance(t, int): t = chr(t) # codepoint
        if self.selection.instance_count == 0:
            self.text = self.text[:self.cursor.idx] + t + self.text[self.cursor.idx:]
            self.update()
            self.cursor.update(self.cursor.idx + len(t))
        else:
            self.text = self.text[:self.selection.start] + t + self.text[self.selection.end:]
            self.update()
            self.cursor.update(self.selection.start + len(t))
            self.selection.reset()

    def erase(self, right=False):
        if self.selection.instance_count == 0:
            if self.cursor.idx == 0 and right == False: return
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
        if key == GLFW_KEY_UP: text.goto(text.cursor.rpos - vec2(0, text.lineheightUsed), allow_drift=False, selection=selection)
        if key == GLFW_KEY_DOWN: text.goto(text.cursor.rpos + vec2(0, text.lineheightUsed), allow_drift=False, selection=selection)
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
        if key == GLFW_KEY_C and mods & GLFW_MOD_CONTROL and text.selection.instance_count > 0: glfwSetClipboardString(window, text.text[text.selection.start:text.selection.end].encode()) # copy
        if key == GLFW_KEY_V and mods & GLFW_MOD_CONTROL: text.write(glfwGetClipboardString(window).decode()) # paste
        if key == GLFW_KEY_X and mods & GLFW_MOD_CONTROL and text.selection.instance_count > 0: # cut
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
        self.glyphAtlas:TextureAtlas
        self.nodes = []
Scene = _Scene()

glfwInit()
glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, True)
window = glfwCreateWindow(Scene.w, Scene.h, Scene.title.encode("utf-8"), None, None)
glfwMakeContextCurrent(window)
Scene.glyphAtlas = TextureAtlas(GL_RED) # # GL_RED because single channel, assign after window creation + assignment else texture settings won't apply

glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)
glfwSetCharCallback(window, char_callback)
glfwSetKeyCallback(window, key_callback)
glfwSetMouseButtonCallback(window, mouse_callback)
glfwSetCursorPosCallback(window, cursor_pos_callback)
glfwSetScrollCallback(window, scroll_callback)

# base unit quad used with instancing
quad_vertices = [
    0.0, 1.0, 0.0, 0.0, 1.0,  # top-left
    1.0, 1.0, 0.0, 1.0, 1.0,  # top-right
    0.0, 0.0, 0.0, 0.0, 0.0,  # bottom-left
    1.0, 0.0, 0.0, 1.0, 0.0   # bottom-right
]
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
quad_instance_data = [0] * 9 # first data is for cursor, later data for selection
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

Scene.fonts["Fira Code"] = Font("assets/fonts/Fira_Code_v6.2/ttf/FiraCode-Regular.ttf")
Scene.font = "Fira Code"

@dataclass(frozen=True)
class Color:
    r:float
    g:float
    b:float
    a:float

def color_from_hex(x:int) -> Color:
    assert x <= 0xffffffff
    if x <= 0xfff: return Color(((x & 0xf00) >> 8) / 15, ((x & 0xf0) >> 4) / 15, (x & 0xf) / 15, 1.0)
    elif x <= 0xffff: return Color(((x & 0xf000) >> 12) / 15, ((x & 0xf00) >> 8) / 15, ((x & 0xf0) >> 4) / 15, (x & 0xf) / 15)
    elif x <= 0xffffff: return Color(((x & 0xff0000) >> 16) / 255, ((x & 0xff00) >> 8) / 255, (x & 0xff) / 255, 1.0)
    else: return Color(((x & 0xff000000) >> 24) / 255, ((x & 0xff0000) >> 16) / 255, ((x & 0xff00) >> 8) / 255, (x & 0xff) / 255)

text_width = 500
text_color = Color(1.0, 0.5, 0.2, 1.0)
with open("text.txt", "r") as f: text = Text(f.read(), (Scene.w-text_width)/2, 0, text_width, Scene.h, text_color)

# from spiritstream.image import Image
# Image.write(list(Scene.glyphAtlas.bitmap), "glyph atlas.bmp")

with open(Path(__file__).parent / "spiritstream/shaders/texquad.frag", "r") as f: fragment_shader_source = f.read()
with open(Path(__file__).parent / "spiritstream/shaders/texquad.vert", "r") as f: vertex_shader_source = f.read()
texquadShader = Shader(vertex_shader_source, fragment_shader_source, ["glyphAtlas", "scale", "offset"])
texquadShader.setUniform("glyphAtlas", 0, "1i")  # 0 means GL_TEXTURE0)

with open(Path(__file__).parent / "spiritstream/shaders/quad.frag", "r") as f: fragment_shader_source = f.read()
with open(Path(__file__).parent / "spiritstream/shaders/quad.vert", "r") as f: vertex_shader_source = f.read()
quadShader = Shader(vertex_shader_source, fragment_shader_source, ["scale", "offset"])

fps = None
frame_count = 0
last_frame_time = time.time()

glEnable(GL_DEPTH_TEST)
glClearDepth(1)
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
        offset_vector = vec2(Scene.x * 2 / Scene.w, Scene.y * 2 / Scene.h) # offset applied to all objects. unit in opengl coordinates
        Scene.resized = False
        glViewport(0, 0, Scene.w, Scene.h)
    if Scene.scrolled:
        offset_vector = vec2(Scene.x * 2 / Scene.w, Scene.y * 2 / Scene.h) # offset applied to all objects. unit in opengl coordinates
        if text.selection.instance_count > 0: text.selection.update()
        Scene.scrolled = False
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    scale, offset = (2 / Scene.w, -2 / Scene.h), tuple((vec2(-1, 1) + offset_vector).components())

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
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, 1+text.selection.instance_count) # only cursor for now
    
    # texture quads
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
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, text.instance_count)

    if (error:=glGetError()) != GL_NO_ERROR: print(f"OpenGL Error: {hex(error)}")

    glfwSwapBuffers(window)
    glfwWaitEvents()

glDeleteBuffers(1, quad_VBO)
glDeleteBuffers(1, quad_EBO)
glDeleteVertexArrays(1, tex_quad_VAO)
glDeleteVertexArrays(1, quad_VAO)
glDeleteBuffers(1, tex_quad_instance_vbo)
glDeleteBuffers(1, quad_instance_vbo)
texquadShader.delete()
glfwTerminate()