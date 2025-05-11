import math, time
from spiritstream.vec import vec2
from spiritstream.bindings.glfw import *
from spiritstream.bindings.opengl import *
from spiritstream.font import Font
from copy import deepcopy

FONTSIZE = 12
LINEHEIGHT = 1.5
DPI = 96
WIDTH = 500
HEIGHT = 1000
PADDING = 5 # px
RESIZED = False

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

with open("text.txt", "r") as f: text = f.read()

# load glyph atlas
chars = set([chr(i) for i in range(32,128)] + list("öäüß"))
F = Font("assets/fonts/Fira_Code_v6.2/ttf/FiraCode-Regular.ttf")
grid_size = math.ceil(math.sqrt(len(chars)))
glyphs = {c: F.render(c, FONTSIZE, DPI) for c in chars}
xMin, xMax = min(g.bearing.x for g in glyphs.values()), max(g.bearing.x + g.size.x for g in glyphs.values())
yMin, yMax = min(g.bearing.y - g.size.y for g in glyphs.values()), max(g.bearing.y for g in glyphs.values())
tile = vec2(xMax - xMin, yMax - yMin) # big enough for every glyph
origin = vec2(-xMin, -yMin)
texture_size = tile * grid_size
bmp = [[0 for x in range(texture_size.x)] for row in range(texture_size.y)] # single channel
for i, v in enumerate(glyphs.values()):
    v.texture_coords = [
        (top_left:=vec2(tile.x * (i % grid_size), tile.y * (i // grid_size)) + origin + v.bearing), # top left
        top_left + vec2(v.size.x, 0), # top right
        top_left + vec2(0, -v.size.y), # bottom left
        top_left + vec2(v.size.x, -v.size.y) # bottom right
    ]
    for y, row in enumerate(reversed(v.bitmap)): # bitmap starts with top row, but the new array is written assuming the first row is the bottom row. OpenGL expects textures that way.
        assert 0 <= (y_idx := v.texture_coords[2].y + y) < texture_size.y, f"{y_idx=}, {texture_size.x=}"
        for x, px in enumerate(row):
            assert 0 <= (x_idx := v.texture_coords[2].x + x) < texture_size.x, f"{x_idx=}, {texture_size.x=}"
            bmp[y_idx][x_idx] = int(px*255) # red channel only
    v.texture_coords = [coord / texture_size for coord in v.texture_coords] # scale texture coordinates for use in vertices
for v in glyphs.values(): del v.bitmap

# create hitboxes
    # normalize tile, texture and glyph data into range 0 to 2
scaled_glyphs, scaled_tile, padding, textbox = {}, None, None, None
def window_norm(v:vec2): return (v/vec2(WIDTH/2, HEIGHT/2))
def rescale():
    global scaled_glyphs, scaled_tile, padding, textbox
    scaled_glyphs, scaled_tile, padding, textbox = {}, None, None, None
    scaled_glyphs = deepcopy(glyphs)
    scaled_tile = window_norm(tile)
    for k, v in glyphs.items():
        scaled_glyphs[k].bearing = window_norm(v.bearing)
        scaled_glyphs[k].size = window_norm(v.size)
        scaled_glyphs[k].advance = v.advance*2/WIDTH
    padding = PADDING * 2 / WIDTH
    textbox = [-1 + padding, 1 - padding, 1 - padding, -1 + padding] # x0, y0, x1, y1
rescale()

cursor = 0 # start at index 0
def getTextQuads():
    """returns tuple(vertices, indices, cursor_coords)"""
    vertices = []
    indices = []
    offset = vec2(textbox[0], textbox[1]-scaled_tile.y) # bottom left corner of a character that is guaranteed to fit vertically in the top row 
    cursor_coords = []
    for char in text:
        cursor_coords.append((offset-vec2(2/WIDTH, 0)).copy()) # move cursor 1 pixel to left of char
        if ord(char) == 10:  # newline
            offset = vec2(textbox[0], offset.y-scaled_tile.y*LINEHEIGHT)
            continue
        g = scaled_glyphs[char]
        if ord(char) == 32: # space
            offset.x += g.advance
            continue
        if offset.x + g.advance > textbox[2]: offset = vec2(textbox[0], offset.y-scaled_tile.y*LINEHEIGHT) # new line, no word splitting
        vertices += [
            # x                                        y                                    z  texture
            (top_left_x := offset.x + g.bearing.x), (top_left_y := offset.y + g.bearing.y), 0, *g.texture_coords[0].components(), # top left
            top_left_x + g.size.x,                  top_left_y,                             0, *g.texture_coords[1].components(), # top right
            top_left_x,                             top_left_y-g.size.y,                    0, *g.texture_coords[2].components(), # bottom left
            top_left_x + g.size.x,                  top_left_y-g.size.y,                    0, *g.texture_coords[3].components()  # bottom right
        ]
        last = len(vertices)//5 - 1 # 5 because xyz and texture xy makes 5 values per vertex
        indices += [
            last-3, last-2, last-1, # triangle top left - top right - bottom left
            last-2, last-1, last # triangle top right - bottom left - bottom right
        ]
        offset.x += g.advance
    cursor_coords.append(offset.copy()) # first position after the last character
    return vertices, indices, cursor_coords
vertices, indices, cursor_coords = getTextQuads()

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

out vec2 texCoord;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
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
loc = glGetUniformLocation(shaderProgram, b"glyphTexture")
glUniform1i(loc, 0)  # 0 means GL_TEXTURE0

# load texture
glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
texture = ctypes.c_uint()
glGenTextures(1, ctypes.byref(texture))
glActiveTexture(GL_TEXTURE0)
glBindTexture(GL_TEXTURE_2D, texture)
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

# TEXT vertex buffer, vertex array and element buffer
VBO, VAO, EBO = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
glGenBuffers(1, ctypes.byref(VBO))
glGenVertexArrays(1, ctypes.byref(VAO))
glGenBuffers(1, ctypes.byref(EBO))
glBindVertexArray(VAO)

glBindBuffer(GL_ARRAY_BUFFER, VBO)
vertices_ctypes = (ctypes.c_float * len(vertices))(*vertices)
glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(vertices_ctypes), vertices_ctypes, GL_DYNAMIC_DRAW)

indices_ctypes = (ctypes.c_uint * len(indices)) (*indices)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(indices_ctypes), indices_ctypes, GL_DYNAMIC_DRAW)

# vertices
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
glEnableVertexAttribArray(0)
# texture coordinate
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(1)

glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindVertexArray(0)

def getCursorQuad(): return [
        # x,y                                                                            z    texture
        *(cursor_coords[cursor] + vec2(0, (g:=scaled_glyphs["|"]).size.y)).components(), 0.2, *g.texture_coords[0].components(), # top left
        *(cursor_coords[cursor] + g.size).components(),                                  0.2, *g.texture_coords[1].components(), # top right
        *cursor_coords[cursor].components(),                                             0.2, *g.texture_coords[2].components(), # bottom left
        *(cursor_coords[cursor] + vec2(g.size.x, 0)).components(),                       0.2, *g.texture_coords[3].components(), # bottom right
    ]
cursor_vertices = cursor_vertices = getCursorQuad()
cursor_indices = [0, 1, 2,   1, 2, 3]

# CURSOR vertex buffer, vertex array and element buffer
VBO1, VAO1, EBO1 = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
glGenBuffers(1, ctypes.byref(VBO1))
glGenVertexArrays(1, ctypes.byref(VAO1))
glGenBuffers(1, ctypes.byref(EBO1))
glBindVertexArray(VAO1)

glBindBuffer(GL_ARRAY_BUFFER, VBO1)
cursor_vertices_ctypes = (ctypes.c_float * len(cursor_vertices))(*cursor_vertices)
glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(cursor_vertices_ctypes), cursor_vertices_ctypes, GL_DYNAMIC_DRAW)

cursor_indices_ctypes = (ctypes.c_uint * len(cursor_indices)) (*cursor_indices)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO1)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(cursor_indices_ctypes), cursor_indices_ctypes, GL_STATIC_DRAW)

# vertices
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
glEnableVertexAttribArray(0)
# texture coordinate
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(1)

def updateCursor(new_position):
    global cursor
    cursor = new_position % len(cursor_coords)
    cursor_vertices = getCursorQuad()
    cursor_vertices_ctypes = (ctypes.c_float * len(cursor_vertices))(*cursor_vertices)
    glBindBuffer(GL_ARRAY_BUFFER, VBO1)
    glBufferSubData(GL_ARRAY_BUFFER, 0, ctypes.sizeof(cursor_vertices_ctypes), cursor_vertices_ctypes)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

def updateText():
    global vertices, indices, cursor_coords
    vertices, indices, cursor_coords = getTextQuads()
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBindVertexArray(VAO)
    vertices_ctypes = (ctypes.c_float * len(vertices))(*vertices)
    glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(vertices_ctypes), vertices_ctypes, GL_DYNAMIC_DRAW)

    indices_ctypes = (ctypes.c_uint * len(indices)) (*indices)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
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

@GLFWcharfun
def char_callback(window, codepoint):
    global text, cursor
    text = text[:cursor] + chr(codepoint) + text[cursor:]
    updateText()
    updateCursor(cursor+1)

@GLFWkeyfun
def key_callback(window, key:int, scancode:int, action:int, mods:int):
    global text, cursor
    if action in [GLFW_PRESS, GLFW_REPEAT]:
        if key == GLFW_KEY_LEFT: updateCursor(cursor-1)
        if key == GLFW_KEY_RIGHT: updateCursor(cursor+1)
        if key == GLFW_KEY_BACKSPACE:
            updateCursor(cursor-1)
            text = text[:cursor] + text[cursor+1:]
            updateText()
        if key == GLFW_KEY_ENTER:
            text = text[:cursor] + "\n" + text[cursor:]
            updateText()
            updateCursor(cursor+1)
        if key == GLFW_KEY_S:
            if mods & GLFW_MOD_CONTROL: # SAVE
                with open("text.txt", "w") as f: f.write(text)

glfwSetCharCallback(window, char_callback)
glfwSetKeyCallback(window, key_callback)

fps = None
frame_count = 0
last_frame_time = time.time()
last_resize_time = time.time()
glClearColor(0, 0, 0, 1)

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
    if RESIZED and current_time - last_resize_time >= 0.2:
        RESIZED = False
        glViewport(0, 0, WIDTH, HEIGHT)
        rescale()
        updateText()
        updateCursor(cursor)
        last_resize_time = current_time

    
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(shaderProgram)
    glUniform3f(glGetUniformLocation(shaderProgram, b"textColor"), 1.0, 0.5, 0.2)
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, 0)

    glUniform3f(glGetUniformLocation(shaderProgram, b"textColor"), 1.0, 1.0, 1.0)
    glBindVertexArray(VAO1)
    glDrawElements(GL_TRIANGLES, len(cursor_indices), GL_UNSIGNED_INT, 0)


    if (error:=glGetError()) != GL_NO_ERROR: print(f"OpenGL Error: {hex(error)}")

    glfwSwapBuffers(window)
    glfwWaitEvents()

glDeleteVertexArrays(1, VAO)
glDeleteBuffers(1, VBO)
glDeleteBuffers(1, EBO)
glDeleteVertexArrays(1, VAO1)
glDeleteBuffers(1, VBO1)
glDeleteBuffers(1, EBO1)
glDeleteProgram(shaderProgram)
glfwTerminate()