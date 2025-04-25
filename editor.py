from ttf import Interpreter, vec2
import math, struct
from glfw import *
from opengl import *

def pad(v, step): return v+(step-v%step)%step

glfwInit()
glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, True)
glfwWindowHint(GLFW_RESIZABLE, False)

width, height = 800, 1000
window = glfwCreateWindow(width, height, b"Spiritstream", None, None)
glfwMakeContextCurrent(window)

with open("text.txt", "r") as f: text = f.read()

# load glyph atlas
linegap = 10 / height # normalized line gap
fontsize = 12
dpi = 96
char_codes = [ord(c) for c in set(text)]
grid_size = math.ceil(math.sqrt(len(char_codes)))
I =  Interpreter(fontfile="assets/fonts/Fira_Code_v6.2/ttf/FiraCode-Regular.ttf")
glyphs = {}
for code in char_codes:
    I.getGlyph(code, fontsize=fontsize, dpi=dpi, antialiasing=3)
    glyphs[code] = {
        "bitmap": I.g.bitmap.copy(),
        "bearing": vec2(I.g.bearing.x, I.g.bearing.y),
        "size": vec2(len(I.g.bitmap[0]) if len(I.g.bitmap) else 0, len(I.g.bitmap)),
        "advance": I.g.advanceWidth
    }
minX, maxX = min([g["bearing"].x for g in glyphs.values()]), max([g["bearing"].x + g["size"].x for g in glyphs.values()])
minY, maxY = min([g["bearing"].y - g["size"].y for g in glyphs.values()]), max([g["bearing"].y for g in glyphs.values()])
tile = vec2(maxX - minX, maxY - minY) # big enough for every glyph
origin = vec2(-minX, minY) # relative to bottom left corner. y is because in array indexing increasing y points down
texture_width = pad(tile.x * grid_size, 4)
texture_height = pad(tile.y * grid_size, 4)
bmp = [[0 for row in range(texture_width)] for x in range(texture_height)] # red channel only
for i, (k, v) in enumerate(glyphs.items()):
    tile_bottom_left_corner = vec2(tile.x * (i % grid_size), tile.y * (i // grid_size + 1))
    bearing = vec2(v["bearing"].x, - v["bearing"].y) # y is inverted because in array indexing increasing y points down
    glyph_top_left_corner = tile_bottom_left_corner + origin + bearing
    glyphs[k]["top_left_corner"] = vec2(glyph_top_left_corner.x, len(bmp) - glyph_top_left_corner.y) # texture coordinates have origin in bottom left corner. this was using origin in top left corner
    for y, row in enumerate(v["bitmap"]):
        bmp_y = glyph_top_left_corner.y + y
        for x, px in enumerate(row):
            bmp_x = glyph_top_left_corner.x + x
            assert bmp_x < texture_width
            assert texture_height > texture_height-bmp_y-1 >= 0, f"{bmp_y}"
            bmp[texture_height-bmp_y-1][bmp_x] = int(px*255) # red channel only and invert y
for v in glyphs.values(): del v["bitmap"]



# create hitboxes
    # normalize tile, texture and glyph data into range 0 to 2. the actual coordinates are -1 to 1, but these are just relative distances
print(texture_width, texture_height)
def window_norm(v:vec2): return (v/vec2(width/2, height/2))
tile = window_norm(tile)
for v in glyphs.values():
    v["top_left_corner"] /= vec2(texture_width, texture_height)
    v["texture_size"] = v["size"] / vec2(texture_width, texture_height)
    v["bearing"] = window_norm(v["bearing"])
    v["size"] = window_norm(v["size"])
    v["advance"] = v["advance"]*2/width
vertices = []
indices = []
offset = vec2(-1, 1-tile.y) # bottom left corner of a character that is guaranteed to fit vertically
for char in text:
    if ord(char) == 10:  # newline
        offset = vec2(-1, offset.y-tile.y-linegap)
        continue
    g = glyphs[ord(char)]
    if ord(char) == 32: # space
        offset.x += g["advance"]
        continue
    if offset.x + g["advance"] > 1: offset = vec2(-1, offset.y-tile.y-linegap) # new line, no word splitting
    vertices += [
        # x                                        y                                          z  texture x                                     texture y 
        (top_left_x := offset.x + g["bearing"].x), (top_left_y := offset.y + g["bearing"].y), 0, g["top_left_corner"].x,                       g["top_left_corner"].y,                       # top left
        top_left_x + g["size"].x,                  top_left_y,                                0, g["top_left_corner"].x + g["texture_size"].x, g["top_left_corner"].y,                       # top right
        top_left_x,                                top_left_y-g["size"].y,                    0, g["top_left_corner"].x,                       g["top_left_corner"].y - g["texture_size"].y, # bottom left
        top_left_x + g["size"].x,                  top_left_y-g["size"].y,                    0, g["top_left_corner"].x + g["texture_size"].x, g["top_left_corner"].y - g["texture_size"].y  # bottom right
    ]
    vertex_group_size = 5 # xyz and texture xy
    last = len(vertices)//vertex_group_size - 1
    indices += [
        last-3, last-2, last-1, # triangle top left - top right - bottom left
        last-2, last-1, last # triangle top right - bottom left - bottom right
    ]
    offset.x += g["advance"]

def compile_shader(shader_type, source_code):
    shader_id = glCreateShader(shader_type)
    src_encoded = source_code.encode('utf-8')
    
    # Single-string variant
    src_ptr = ctypes.c_char_p(src_encoded)
    src_array = (ctypes.c_char_p * 1)(src_ptr)
    length_array = (ctypes.c_int * 1)(len(src_encoded))
    
    glShaderSource(shader_id, 1, src_array, length_array)
    glCompileShader(shader_id)
    
    # Error checking
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
out vec4 FragColor;

in vec2 texCoord;
uniform sampler2D glyphTexture;

void main()
{
    FragColor = vec4(1.0f, 0.5f, 0.2f, texture(glyphTexture, texCoord).r);
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
texture = ctypes.c_uint()
glGenTextures(1, ctypes.byref(texture))
glActiveTexture(GL_TEXTURE0)
glBindTexture(GL_TEXTURE_2D, texture)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
flat_bmp = [x for y in bmp for x in y]
assert len(flat_bmp) == texture_width * texture_height
bmp_ctypes = (ctypes.c_ubyte * len(flat_bmp))(*flat_bmp)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, texture_width, texture_height, 0, GL_RED, GL_UNSIGNED_BYTE, bmp_ctypes)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# vertex buffer, vertex array and element buffer
VBO, VAO, EBO = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
glGenBuffers(1, ctypes.byref(VBO))
glGenVertexArrays(1, ctypes.byref(VAO))
glGenBuffers(1, ctypes.byref(EBO))
glBindVertexArray(VAO)

glBindBuffer(GL_ARRAY_BUFFER, VBO)
vertices_ctypes = (ctypes.c_float * len(vertices))(*vertices)
glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(vertices_ctypes), vertices_ctypes, GL_STATIC_DRAW)

indices_ctypes = (ctypes.c_uint * len(indices)) (*indices)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, ctypes.sizeof(indices_ctypes), indices_ctypes, GL_STATIC_DRAW)

# vertices
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
glEnableVertexAttribArray(0)
# texture coordinate
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3*ctypes.sizeof(ctypes.c_float)))
glEnableVertexAttribArray(1)

glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindVertexArray(0)

while not glfwWindowShouldClose(window):
    if glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS: glfwSetWindowShouldClose(window, GLFW_TRUE)
    glClearColor(0.3, 0.2, 0.2, 1)
    glClear(GL_COLOR_BUFFER_BIT)

    glUseProgram(shaderProgram)
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, 0)

    if (error:=glGetError()) != GL_NO_ERROR: print(f"OpenGL Error: {hex(error)}")

    glfwSwapBuffers(window)
    glfwPollEvents()

glDeleteVertexArrays(1, VAO)
glDeleteBuffers(1, VBO)
glDeleteBuffers(1, EBO)
glDeleteProgram(shaderProgram)

glfwTerminate()