import math, time
from spiritstream.vec import vec2
from spiritstream.bindings.glfw import *
from spiritstream.bindings.opengl import *
from spiritstream.font import Font
from typing import Union

FONTSIZE = 12
LINEHEIGHT = 1.5
DPI = 96
WIDTH = 700
HEIGHT = 1000
PADDING = 5 # px
RESIZED = True

glfwInit()
glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, True)
window = glfwCreateWindow(WIDTH, HEIGHT, b"Spiritstream", None, None)
glfwMakeContextCurrent(window)

@GLFWframebuffersizefun
def framebuffer_size_callback(window, width, height):
    global WIDTH, HEIGHT, RESIZED
    WIDTH, HEIGHT, RESIZED = width, height, True
glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)

class GlyphAtlas:
    def __init__(self, chars:set, F:Font, size:int, dpi:int):
        self.glyphs = {c: F.render(c, FONTSIZE, DPI) for c in chars}

        grid_size = math.ceil(math.sqrt(len(chars)))
        xMin, xMax = min(g.bearing.x for g in self.glyphs.values()), max(g.bearing.x + g.size.x for g in self.glyphs.values())
        yMin, yMax = min(g.bearing.y - g.size.y for g in self.glyphs.values()), max(g.bearing.y for g in self.glyphs.values())

        self.tile = vec2(xMax - xMin, yMax - yMin) # big enough for every glyph

        texture_size = self.tile * grid_size
        bmp = [[0 for x in range(texture_size.x)] for row in range(texture_size.y)] # single channel
        origin = vec2(-xMin, -yMin)
        for i, v in enumerate(self.glyphs.values()):
            v.texture_coords = [
                (top_left:=vec2(self.tile.x * (i % grid_size), self.tile.y * (i // grid_size)) + origin + v.bearing), # top left
                top_left + vec2(v.size.x,                 0), # top right
                top_left + vec2(0,                        -v.size.y), # bottom left
                top_left + vec2(v.size.x,                 -v.size.y) # bottom right
            ]
            for y, row in enumerate(reversed(v.bitmap)): # bitmap starts with top row, but the new array is written assuming the first row is the bottom row. OpenGL expects textures that way.
                assert 0 <= (y_idx := v.texture_coords[2].y + y) < texture_size.y
                for x, px in enumerate(row):
                    assert 0 <= (x_idx := v.texture_coords[2].x + x) < texture_size.x
                    bmp[y_idx][x_idx] = int(px*255) # red channel only
            v.texture_coords = [coord / texture_size for coord in v.texture_coords] # scale texture coordinates for use in vertices
        for v in self.glyphs.values(): del v.bitmap

        # texture
        self.texture = ctypes.c_uint()
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glGenTextures(1, ctypes.byref(self.texture))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        flat_bmp = [x for y in bmp for x in y]
        assert len(flat_bmp) == texture_size.x * texture_size.y
        bmp_ctypes = (ctypes.c_ubyte * len(flat_bmp))(*flat_bmp)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, texture_size.x, texture_size.y, 0, GL_RED, GL_UNSIGNED_BYTE, bmp_ctypes)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, 0)

    def __getitem__(self, i): return self.glyphs[i]

class Cursor:
    def __init__(self, text:"Text"):
        self.text = text # keep reference for cursor_coords and linewraps
        self.left = None
        self.pos = vec2()
        self.idx = None
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

        self.indices = [0, 1, 2,   1, 2, 3]
        cursor_indices_ctypes = (ctypes.c_uint * len(self.indices)) (*self.indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(cursor_indices_ctypes), cursor_indices_ctypes, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        self.update(0)

    def update(self, pos:Union[int, vec2], allow_drift=True, left=False):
        assert isinstance(left, bool) and isinstance(allow_drift, bool)
        assert isinstance(pos, (int, vec2)), f"Wrong argument type \"pos\": {type(pos)}. Can only use integer (index of char in the text) or vec2 (screen coordinate)"
        linewraps, cursorcoords = self.text.linewraps, self.text.cursorcoords
        if isinstance(pos, vec2): # get idx from pos
            idx = 0
            if not allow_drift: pos.x = self.x
            if pos.y > self.text.cursorcoords[-1].y + (self.text.atlas.tile.y * LINEHEIGHT) / 2: idx, allow_drift = len(self.text.cursorcoords) - 1, True
            elif pos.y < self.text.cursorcoords[0].y - (self.text.atlas.tile.y * LINEHEIGHT) / 2: idx, allow_drift = 0, True
            else: 
                closest = vec2(math.inf, math.inf)
                for i, c in enumerate(self.text.cursorcoords):
                    if (d:=(pos-c).abs()).y <= closest.y:
                        if d.x < closest.x or d.y < closest.y: idx, closest, left = i, d.copy(), False
                        # if this cursor position can be either up or down, check which is closer and take that one. if I take the down one, set left to True
                        if i in linewraps and i < len(self.text.cursorcoords) - 1:
                            d0 = (pos - vec2(2, self.text.cursorcoords[i+1].y)).abs()
                            if d0.y <= closest.y:
                                if d0.y < closest.y or d0.x < closest.x: idx, closest, left = i, d0.copy(), True
                    else: break
        else: idx = pos % len(cursorcoords)
        if idx == self.idx: return
        self.idx = idx
        self.left = left
        self.pos = vec2(0, cursorcoords[(self.idx + 1) % len(cursorcoords)].y) if left else cursorcoords[self.idx]
        g = self.text.atlas["|"]
        vertices = [
            # x,y                                            z    texture
            *(self.pos - vec2(0, g.size.y)).components(),    0.2, *g.texture_coords[0].components(), # top left
            *(self.pos + g.size * vec2(1, -1)).components(), 0.2, *g.texture_coords[1].components(), # top right
            *self.pos.components(),                          0.2, *g.texture_coords[2].components(), # bottom left
            *(self.pos + vec2(g.size.x, 0)).components(),    0.2, *g.texture_coords[3].components(), # bottom right
        ]
        if allow_drift: self.x = self.pos.x

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        vertices_ctypes = (ctypes.c_float * len(vertices))(*vertices)
        assert ctypes.sizeof(vertices_ctypes) == 80, "This size is this hard coded in the preallocation"
        glBufferSubData(GL_ARRAY_BUFFER, 0, 80, vertices_ctypes)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

class Text:
    def __init__(self, text:str, atlas:GlyphAtlas):
        self.text = text
        self.atlas = atlas
        self.linewraps = []
        self.cursorcoords = []
        self.width = 500
        self.x = None # x offset from left border. set on load and resize of the window

        self.VBO, self.VAO, self.EBO = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        glGenBuffers(1, ctypes.byref(self.VBO))
        glGenVertexArrays(1, ctypes.byref(self.VAO))
        glGenBuffers(1, ctypes.byref(self.EBO))

        self.update()
        self.cursor = Cursor(self)

    def update(self):
        vertices = []
        indices = []
        linewraps = []
        offset = vec2(0, self.atlas.tile.y) # bottom left corner of a character that is guaranteed to fit vertically in the top row 
        cursorcoords = []
        for char in self.text:
            cursorcoords.append((offset+vec2(2/WIDTH, 0)).copy()) # move cursor 1 pixel to left of char
            if ord(char) == 10:  # newline
                offset = vec2(0, offset.y+self.atlas.tile.y*LINEHEIGHT)
                continue
            g = self.atlas[char]
            if ord(char) == 32: # space
                offset.x += g.advance
                continue
            if offset.x + g.advance > self.width:
                linewraps.append(len(cursorcoords) - 1) # cursor before this char
                offset = vec2(0, offset.y+self.atlas.tile.y*LINEHEIGHT) # new line, no word splitting
            vertices += [
                # x                                        y                                    z  texture
                (top_left_x := offset.x + g.bearing.x), (top_left_y := offset.y - g.bearing.y), 0, *g.texture_coords[0].components(), # top left
                top_left_x + g.size.x,                  top_left_y,                             0, *g.texture_coords[1].components(), # top right
                top_left_x,                             top_left_y+g.size.y,                    0, *g.texture_coords[2].components(), # bottom left
                top_left_x + g.size.x,                  top_left_y+g.size.y,                    0, *g.texture_coords[3].components()  # bottom right
            ]
            last = len(vertices)//5 - 1 # 5 because xyz and texture xy makes 5 values per vertex
            indices += [
                last-3, last-2, last-1, # triangle top left - top right - bottom left
                last-2, last-1, last # triangle top right - bottom left - bottom right
            ]
            offset.x += g.advance
        cursorcoords.append(offset.copy()) # first position after the last character
        self.cursorcoords, self.indices, self.linewraps = cursorcoords, indices, linewraps

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

    def write(self, codepoint:int, left=False):
        self.text = self.text[:self.cursor.idx] + chr(codepoint) + self.text[self.cursor.idx:]
        self.update()
        self.cursor.update(self.cursor.idx + 1, left=left)

    def erase(self):
        self.text = self.text[:self.cursor.idx-1] + self.text[self.cursor.idx:]
        self.update()
        self.cursor.update(self.cursor.idx - 1)
        # self.cursor.update(self.cursor.idx - 1, left=self.cursor.idx-1 in self.linewraps or self.text[self.cursor.idx - 1] == "\n")

glyph_atlas = GlyphAtlas(set([chr(i) for i in range(32,128)] + list("öäüß")), Font("assets/fonts/Fira_Code_v6.2/ttf/FiraCode-Regular.ttf"), FONTSIZE, DPI)
with open("text.txt", "r") as f: text = Text(f.read(), glyph_atlas)

def compile_shader(shader_type, source_code):
    shader_id = glCreateShader(shader_type)
    src_encoded = source_code.encode('utf-8')
    
    src_ptr = ctypes.c_char_p(src_encoded)
    src_array = (ctypes.c_char_p * 1)(src_ptr)
    length_array = (ctypes.c_int * 1)(len(src_encoded))
    
    glShaderSource(shader_id, 1, src_array, length_array)
    glCompileShader(shader_id)
    
    success = ctypes.c_int(0)
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, ctypes.byref(success))
    if not success.value:
        error_log = ctypes.create_string_buffer(512)
        glGetShaderInfoLog(shader_id, 512, None, error_log)
        raise RuntimeError(f"Shader compilation failed: {error_log.value.decode()}")
    
    return shader_id

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

vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_shader_source)

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
fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_source)

shaderProgram = glCreateProgram()
glAttachShader(shaderProgram, vertex_shader)
glAttachShader(shaderProgram, fragment_shader)
glLinkProgram(shaderProgram)
success = ctypes.c_int(0)
glGetProgramiv(shaderProgram, GL_LINK_STATUS, ctypes.byref(success))
if not success.value:
    error_log = ctypes.create_string_buffer(512)
    glGetProgramInfoLog(shaderProgram, 512, None, error_log)
    raise RuntimeError(f"Program link failed: {error_log.value.decode()}")
glUseProgram(shaderProgram)
glDeleteShader(vertex_shader)
glDeleteShader(fragment_shader)
texture_loc = glGetUniformLocation(shaderProgram, b"glyphTexture")
glUniform1i(texture_loc, 0)  # 0 means GL_TEXTURE0
textcolor_loc = glGetUniformLocation(shaderProgram, b"textColor")
scale_loc = glGetUniformLocation(shaderProgram, b"scale")
offset_loc = glGetUniformLocation(shaderProgram, b"offset")

@GLFWcharfun
def char_callback(window, codepoint): text.write(codepoint)

@GLFWkeyfun
def key_callback(window, key:int, scancode:int, action:int, mods:int):
    global text, cursor
    if action in [GLFW_PRESS, GLFW_REPEAT]:
        if key == GLFW_KEY_LEFT: text.cursor.update(text.cursor.idx - 1, left=text.cursor.idx-1 in text.linewraps)
        if key == GLFW_KEY_RIGHT: text.cursor.update(text.cursor.idx+1)
        if key == GLFW_KEY_UP: text.cursor.update(text.cursor.pos - vec2(0, text.atlas.tile.y * LINEHEIGHT), allow_drift=False, left=text.cursor.idx in text.linewraps)
        if key == GLFW_KEY_DOWN: text.cursor.update(text.cursor.pos + vec2(0, text.atlas.tile.y * LINEHEIGHT), allow_drift=False, left=text.cursor.idx in text.linewraps)
        if key == GLFW_KEY_BACKSPACE: text.erase()
        if key == GLFW_KEY_ENTER: text.write(ord("\n"), left=True)
        if key == GLFW_KEY_S:
            if mods & GLFW_MOD_CONTROL: # SAVE
                with open("text.txt", "w") as f: f.write(text.text)
        if key == GLFW_KEY_HOME: text.cursor.update(min([0] + [i for i, c in enumerate(text.text) if c == "\n"] + text.linewraps, key=lambda x: text.cursor.idx - x if x < text.cursor.idx or (x == text.cursor.idx and text.cursor.left) else math.inf), left=True)
        if key == GLFW_KEY_END: text.cursor.update(min([0] + [i for i, c in enumerate(text.text) if c == "\n"] + text.linewraps, key=lambda x: x - text.cursor.idx if x > text.cursor.idx or (x == text.cursor.idx != 0 and not text.cursor.left) else math.inf))

@GLFWmousebuttonfun
def mouse_callback(window, button:int, action:int, mods:int):
    if button == GLFW_MOUSE_BUTTON_1 and action == GLFW_PRESS:
        x, y = ctypes.c_double(), ctypes.c_double()
        glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
        pos = vec2(x.value - text.x, y.value) # adjust for offset vector
        text.cursor.update(pos)

glfwSetCharCallback(window, char_callback)
glfwSetKeyCallback(window, key_callback)
glfwSetMouseButtonCallback(window, mouse_callback)

fps = None
frame_count = 0
last_frame_time = time.time()

glClearColor(0, 0, 0, 1)
glBindTexture(GL_TEXTURE_2D, text.atlas.texture)

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
    if RESIZED:
        text.x = (WIDTH - text.width) / 2
        offset_vector = vec2((text.x) * 2 / WIDTH, 0) # offset applied to all objects. unit in opengl coordinates
        RESIZED = False
        glViewport(0, 0, WIDTH, HEIGHT)
    
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(shaderProgram)
    glUniform2f(scale_loc, 2 / WIDTH, -2 / HEIGHT) # inverted y axis. from my view (0,0) is the top left corner, like in browsers
    glUniform2f(offset_loc, *(vec2(-1, 1) + offset_vector).components())
    glUniform3f(textcolor_loc, 1.0, 0.5, 0.2)
    glBindVertexArray(text.VAO)
    glDrawElements(GL_TRIANGLES, len(text.indices), GL_UNSIGNED_INT, 0)

    glUniform3f(textcolor_loc, 1.0, 1.0, 1.0)
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
glDeleteProgram(shaderProgram)
glfwTerminate()