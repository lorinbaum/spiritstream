"""
- 3 squares that I can move around
- toolbar with background and move, pan, zoom tools
- use file argument and workspace to open file and drag open boxes and save their position and camera position
"""
from spiritstream.bindings.opengl import *
from spiritstream.bindings.glfw import *
from spiritstream.shader import Shader
from spiritstream.buffer import BaseQuad, QuadBuffer, TexQuadBuffer
from spiritstream.tree import color_from_hex
from spiritstream.vec import vec2
import spiritstream.image as Image
import json, argparse

argparser = argparse.ArgumentParser(description="Spiritstream")
argparser.add_argument("file", type=str, nargs="?", default=None, help="Path to file to open")
args = argparser.parse_args()

WORKSPACEPATH = p if (p:=Path(__file__).parent / "cache/workspace.json").exists() else None
if args.file: data = {"last_open": Path(args.file)}
else:
    try:
        with open(WORKSPACEPATH, "r") as f: data = json.load(f)
    except: data = {}
FILEPATH = Path(data.get("last_open", Path(__file__).parent/"Unnamed.canvas"))
X, Y, SCALE = data.get("x", 0), data.get("y", 0), data.get("scale", 1)


WIDTH = 1000
HEIGHT = 1000
RESIZED = True

quads = set()
DRAGGING = None
DRAG_START = None
X_START = Y_START = None
SCALE_START = None
QUADS_UPDATED = True
UI_SELECTION_QUADS_UPDATED = True

SELECTION_COLOR = color_from_hex(0x423024)

ui_elements = set()
ui_selected = None

class Quad:
    def __init__(self, buffer:QuadBuffer, x, y, z, width, height, red, green, blue, alpha=1.0):
        self.buffer, self.x, self.y, self.z, self.w, self.h, self.r, self.g, self.b, self.a = buffer, x, y, z, width, height, red, green, blue, alpha
        self.update()
    def __setattr__(self, name, value):
        global QUADS_UPDATED
        QUADS_UPDATED = True
        super().__setattr__(name, value)
    def update(self):
        self.buffer.add([self.x, self.y, self.z, self.w, self.h, self.r, self.g, self.b, self.a])
    def __repr__(self): return f"Quad(x={self.x}, y={self.y}, w={self.w}, h={self.h}, RGBA=({self.r},{self.g},{self.b},{self.a}))"

class Button:
    def __init__(self, x, y, w, h, icon_path:Path):
        self.x, self.y, self.w, self.h = x, y, w, h
        img = Image.read(icon_path)
        texture = ctypes.c_uint()
        glGenTextures(1, ctypes.byref(texture))
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.data)
        self.tex_quad = TexQuadBuffer(base_quad, texture, img_shader)
        self.tex_quad.add([x+4, y+4, 0, w-8, h-8])

        self.quad = Quad(ui_selected_quad, x, y, 0.1, w, h, (c:=SELECTION_COLOR).r, c.g, c.b, c.a)
        ui_elements.add(self)
    
    def select(self):
        global ui_selected, UI_SELECTION_QUADS_UPDATED
        ui_selected = self
        UI_SELECTION_QUADS_UPDATED = True

@GLFWframebuffersizefun
def framebuffer_size_callback(window, width, height):
    global WIDTH, HEIGHT, RESIZED
    WIDTH, HEIGHT, RESIZED = width, height, True

@GLFWkeyfun
def key_callback(window, key:int, scancode:int, action:int, mods:int):
    if action in [GLFW_PRESS, GLFW_REPEAT]:
        ctrl = bool(mods & GLFW_MOD_CONTROL)
        if key == GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GLFW_TRUE)
        if key == GLFW_KEY_S and ctrl: # save
            data = {"boxes": [{ "x": q.x, "y": q.y, "z": q.z, "w": q.w, "h": q.h, "r": q.r, "g": q.g, "b": q.b, "a": q.a } for q in quads]}
            with open(FILEPATH, "w") as f: json.dump(data, f, indent=2)
            glfwSetWindowTitle(window, (FILEPATH.name).encode())

@GLFWcursorposfun
def cursor_pos_callback(window, x, y):
    global DRAGGING, DRAG_START, X, Y, SCALE, X_START, Y_START
    if glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS:
        x, y = ctypes.c_double(), ctypes.c_double()
        glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
        x, y = x.value, y.value
        drag_offset = vec2(x - DRAG_START.x, y - DRAG_START.y)
        if ui_selected is move_button and DRAGGING is not None:
            DRAGGING.x += drag_offset.x / SCALE
            DRAGGING.y += drag_offset.y / SCALE
            DRAG_START = vec2(x, y)
        elif ui_selected is pan_button:
            X += drag_offset.x
            Y += drag_offset.y
            DRAG_START = vec2(x, y)
        elif ui_selected is zoom_button:
            new_scale = 1.002**(drag_offset.x)
            SCALE = SCALE_START * new_scale
            X = DRAG_START.x + new_scale * (X_START-DRAG_START.x)
            Y = DRAG_START.y + new_scale * (Y_START-DRAG_START.y)
        elif ui_selected is box_button:
            if DRAGGING is None:
                new_quad = Quad(quad, (DRAG_START.x-X)/SCALE, (DRAG_START.y-Y)/SCALE, 0.5, 0, 0, 1.0, 0.2, 0.8) # new quad, always magentaish
                quads.add(new_quad)
                DRAGGING = new_quad
            DRAGGING.w = abs(drag_offset.x / SCALE)
            DRAGGING.h = abs(drag_offset.y / SCALE)
            if x < DRAG_START.x: DRAGGING.x = (x-X) / SCALE
            if y < DRAG_START.y: DRAGGING.y = (y-Y) / SCALE

@GLFWmousebuttonfun
def mouse_callback(window, button:int, action:int, mods:int):
    if button == GLFW_MOUSE_BUTTON_1:
        global DRAGGING, DRAG_START, X, Y, SCALE, SCALE_START, X_START, Y_START
        if action is GLFW_PRESS:
            x, y = ctypes.c_double(), ctypes.c_double()
            glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
            DRAG_START = vec2(x.value, y.value)
            SCALE_START = SCALE
            X_START, Y_START = X, Y
            DRAGGING = next((q for q in quads if q.x<=(x.value-X)/SCALE<=q.x+q.w and q.y<=(y.value-Y)/SCALE<=q.y+q.h), None)
            for e in ui_elements:
                if e.x<=x.value<=e.x+e.w and e.y<=y.value<=e.y+e.h: e.select()
        elif action == GLFW_RELEASE: DRAGGING = DRAG_START = SCALE_START = X_START = Y_START = None

glfwInit()
glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, True)
window = glfwCreateWindow(WIDTH, HEIGHT, "Canvas".encode("utf-8"), None, None)
glfwMakeContextCurrent(window)

glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)
glfwSetKeyCallback(window, key_callback)
glfwSetCursorPosCallback(window, cursor_pos_callback)
glfwSetMouseButtonCallback(window, mouse_callback)

glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glEnable(GL_DEPTH_TEST)
glClearDepth(1)
glClearColor(0, 0, 0, 1)

glfwSetWindowTitle(window, (FILEPATH.name).encode())

quad_shader = Shader(p:=Path(__file__).parent / "spiritstream/shaders/quad.vert", p.parent / "quad.frag", ["scale", "offset"])
img_shader = Shader(p:=Path(__file__).parent / "spiritstream/shaders/img.vert", p.parent / "img.frag", ["scale", "offset"])

base_quad = BaseQuad()
quad = QuadBuffer(base_quad, quad_shader)
ui_quad = QuadBuffer(base_quad, quad_shader) # for static ui
ui_selected_quad = QuadBuffer(base_quad, quad_shader) # for selection

# button background:
Quad(ui_quad, 0, 0, 0.2, 160, 40, 0.1, 0.1, 0.1)

move_button = Button(0, 0, 40, 40, Path(__file__).parent / "assets/images/move_tool.png")
pan_button = Button(40, 0, 40, 40, Path(__file__).parent / "assets/images/pan_tool.png")
zoom_button = Button(80, 0, 40, 40, Path(__file__).parent / "assets/images/zoom_tool.png")
box_button = Button(120, 0, 40, 40, Path(__file__).parent / "assets/images/box_tool.png")

if FILEPATH.exists():
    with open(FILEPATH, "r") as f: data = json.load(f)
    for box in data["boxes"]: quads.add(Quad(quad, box["x"], box["y"], box["z"], box["w"], box["h"], box["r"], box["g"], box["b"], box["a"]))

first_setup = True
while not glfwWindowShouldClose(window):

    if QUADS_UPDATED:
        quad.clear()
        for q in quads: q.update()
        QUADS_UPDATED = False
        if first_setup: first_setup = False
        else: glfwSetWindowTitle(window, (FILEPATH.name + "*").encode())

    if UI_SELECTION_QUADS_UPDATED:
        ui_selected_quad.clear()
        if ui_selected is not None: ui_selected.quad.update()
        UI_SELECTION_QUADS_UPDATED = False

    if RESIZED:
        glViewport(0, 0, WIDTH, HEIGHT)
        RESIZED = False

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    scale, offset = (2 / WIDTH * SCALE, -2 * SCALE / HEIGHT), (-1 + X * 2 / (WIDTH), 1 + Y * -2 / (HEIGHT))
    quad.draw(scale, offset)
    ui_scale, ui_offset = (2 / WIDTH, -2 / HEIGHT), (-1, 1)
    ui_quad.draw(ui_scale, ui_offset)
    ui_selected_quad.draw(ui_scale, ui_offset)
    for e in ui_elements: e.tex_quad.draw(ui_scale, ui_offset)

    glfwSwapBuffers(window)
    glfwPollEvents()

quad_shader.delete()
img_shader.delete()

base_quad.delete()
quad.delete()
for e in ui_elements: e.tex_quad.delete()

if WORKSPACEPATH is not None:
    with open(WORKSPACEPATH, "w") as f: json.dump({"last_open": FILEPATH.as_posix(), "x":X, "y":Y, "scale":SCALE}, f, indent=2)

glfwTerminate()
