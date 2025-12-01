"""
- 3 squares that I can move around
- toolbar with background and move, pan, zoom tools
- use file argument and workspace to open file and drag open boxes and save their position and camera position
- select and rescale individual boxes with the move tool
"""
from spiritstream.bindings.opengl import *
from spiritstream.bindings.glfw import *
from spiritstream.shader import Shader
from spiritstream.buffer import BaseQuad, QuadBuffer, TexQuadBuffer
from spiritstream.tree import color_from_hex
from spiritstream.vec import vec2
import spiritstream.image as Image
import json, argparse
from enum import Enum, auto

argparser = argparse.ArgumentParser(description="Spiritstream")
argparser.add_argument("file", type=str, nargs="?", default=None, help="Path to file to open")
args = argparser.parse_args()

WORKSPACEPATH = p if (p:=Path(__file__).parent / "cache/workspace.json").exists() else None
if args.file: data = {"last_open": Path(args.file)}
else:
    try:
        with open(WORKSPACEPATH, "r") as f: data = json.load(f)
        if data.get("last_open", None) is not None and not Path(data["last_open"]).exists(): data.pop("last_open")
    except: data = {}
FILEPATH = Path(data.get("last_open", Path(__file__).parent/"Unnamed.canvas"))
X, Y, SCALE = data.get("x", 0), data.get("y", 0), data.get("scale", 1)


WIDTH = 1000
HEIGHT = 1000
RESIZED = True

class Dir(Enum):
    TOP_LEFT = auto(); TOP = auto(); TOP_RIGHT = auto()
    LEFT = auto(); RIGHT = auto()
    BOTTOM_LEFT = auto(); BOTTOM = auto(); BOTTOM_RIGHT=auto()

boxes = set()
RESIZING:bool = False # whether currently selected box is being resized or not
RESIZING_POS:None|Dir = None # from which direction of the box it is being scaled
RESIZING_START_X = RESIZING_START_Y = RESIZING_START_W = RESIZING_START_H = None
DRAGGING = None
DRAG_START = None
X_START = Y_START = None
SCALE_START = None
QUADS_UPDATED = True
UI_SELECTION_QUADS_UPDATED = True

SELECTION_COLOR = color_from_hex(0x423024)

ui_elements = set()
ui_selected = None
box_selected = None

class Box:
    def __init__(self, buffer:QuadBuffer, x, y, z, width, height, red, green, blue, alpha=1.0):
        self.buffer, self.x, self.y, self.z, self.w, self.h, self.r, self.g, self.b, self.a = buffer, x, y, z, width, height, red, green, blue, alpha
        self.update()
    def __setattr__(self, name, value):
        global QUADS_UPDATED
        QUADS_UPDATED = True
        super().__setattr__(name, value)
    def update(self):
        self.buffer.add([self.x, self.y, self.z, self.w, self.h, self.r, self.g, self.b, self.a])
    def __repr__(self): return f"Box(x={self.x}, y={self.y}, w={self.w}, h={self.h}, RGBA=({self.r},{self.g},{self.b},{self.a}))"

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

        self.quad = Box(ui_selected_quad, x, y, 0.1, w, h, (c:=SELECTION_COLOR).r, c.g, c.b, c.a)
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
            data = {"boxes": [{ "x": b.x, "y": b.y, "z": b.z, "w": b.w, "h": b.h, "r": b.r, "g": b.g, "b": b.b, "a": b.a } for b in boxes]}
            with open(FILEPATH, "w") as f: json.dump(data, f, indent=2)
            if WORKSPACEPATH is not None:
                with open(WORKSPACEPATH, "w") as f: json.dump({"last_open": FILEPATH.as_posix(), "x":X, "y":Y, "scale":SCALE}, f, indent=2)
            glfwSetWindowTitle(window, (FILEPATH.name).encode())

@GLFWcursorposfun
def cursor_pos_callback(window, x, y):
    global DRAGGING, DRAG_START, X, Y, SCALE, X_START, Y_START
    if glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS:
        x, y = ctypes.c_double(), ctypes.c_double()
        glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
        x, y = x.value, y.value
        drag_offset = vec2(x - DRAG_START.x, y - DRAG_START.y)
        if ui_selected is move_button:
            if RESIZING is True:
                global RESIZING_POS, RESIZING_START_X, RESIZING_START_Y, RESIZING_START_W, RESIZING_START_H, box_selected
                assert all((n is not None for n in (RESIZING_POS, RESIZING_START_X, RESIZING_START_Y, RESIZING_START_W, RESIZING_START_H)))
                right_side, bot_side = RESIZING_START_X+RESIZING_START_W, RESIZING_START_Y+RESIZING_START_H
                # X
                if RESIZING_POS in (Dir.TOP_LEFT, Dir.LEFT, Dir.BOTTOM_LEFT): box_selected.x = min(right_side, RESIZING_START_X+drag_offset.x/SCALE)
                elif RESIZING_POS in (Dir.TOP_RIGHT, Dir.RIGHT, Dir.BOTTOM_RIGHT): box_selected.x = min(RESIZING_START_X, right_side+drag_offset.x/SCALE)
                # Y
                if RESIZING_POS in (Dir.TOP_LEFT, Dir.TOP, Dir.TOP_RIGHT): box_selected.y = min(bot_side, RESIZING_START_Y+drag_offset.y/SCALE)
                elif RESIZING_POS in (Dir.BOTTOM_LEFT, Dir.BOTTOM, Dir.BOTTOM_RIGHT): box_selected.y = min(RESIZING_START_Y, bot_side+drag_offset.y/SCALE)
                # W
                if RESIZING_POS in (Dir.TOP_LEFT, Dir.LEFT, Dir.BOTTOM_LEFT): box_selected.w = abs(RESIZING_START_W-drag_offset.x/SCALE) 
                elif RESIZING_POS in (Dir.TOP_RIGHT, Dir.RIGHT, Dir.BOTTOM_RIGHT): box_selected.w = abs(RESIZING_START_W+drag_offset.x/SCALE)
                # H
                if RESIZING_POS in (Dir.TOP_LEFT, Dir.TOP, Dir.TOP_RIGHT): box_selected.h = abs(RESIZING_START_H-drag_offset.y/SCALE)
                elif RESIZING_POS in (Dir.BOTTOM_LEFT, Dir.BOTTOM, Dir.BOTTOM_RIGHT): box_selected.h = abs(RESIZING_START_H+drag_offset.y/SCALE)                
                
                box_selected_quad.replace([box_selected.x, box_selected.y, 0, box_selected.w, box_selected.h, (c:=SELECTION_COLOR).r, c.g, c.b, 0.5])
                    
            elif DRAGGING is not None:
                DRAGGING.x += drag_offset.x / SCALE
                DRAGGING.y += drag_offset.y / SCALE
                box_selected_quad.replace([DRAGGING.x, DRAGGING.y, 0, DRAGGING.w, DRAGGING.h, (c:=SELECTION_COLOR).r, c.g, c.b, 0.5])
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
                new_box = Box(quad, (DRAG_START.x-X)/SCALE, (DRAG_START.y-Y)/SCALE, 0.5, 0, 0, 1.0, 0.2, 0.8) # new quad, always magentaish
                boxes.add(new_box)
                DRAGGING = new_box
            DRAGGING.w = abs(drag_offset.x / SCALE)
            DRAGGING.h = abs(drag_offset.y / SCALE)
            if x < DRAG_START.x: DRAGGING.x = (x-X) / SCALE
            if y < DRAG_START.y: DRAGGING.y = (y-Y) / SCALE
    else:
        if box_selected is not None:
            # hover effects for resizing selected box
            sx, sy = (x-X)/SCALE, (y-Y)/SCALE
            tol = 10 / SCALE
            if box_selected.x-tol <= sx <= box_selected.x+box_selected.w+tol and box_selected.y-tol <= sy <= box_selected.y+box_selected.h+tol:
                top, bottom = sy<=box_selected.y, sy >= box_selected.y+box_selected.h
                if sx<=box_selected.x: # left
                    if top: glfwSetCursor(window, cursor_nwse)
                    elif bottom: glfwSetCursor(window, cursor_nesw)
                    else: glfwSetCursor(window, cursor_ew)
                elif sx >= box_selected.x+box_selected.w: # right
                    if top: glfwSetCursor(window, cursor_nesw)
                    elif bottom: glfwSetCursor(window, cursor_nwse)
                    else: glfwSetCursor(window, cursor_ew)
                elif top or bottom: glfwSetCursor(window, cursor_ns)
                else: glfwSetCursor(window, cursor_arrow)
            else: glfwSetCursor(window, cursor_arrow)

@GLFWmousebuttonfun
def mouse_callback(window, button:int, action:int, mods:int):
    if button == GLFW_MOUSE_BUTTON_1:
        global DRAGGING, DRAG_START, X, Y, SCALE, SCALE_START, X_START, Y_START, box_selected, RESIZING, RESIZING_POS, RESIZING_START_X,\
            RESIZING_START_Y, RESIZING_START_W, RESIZING_START_H
        if action is GLFW_PRESS:
            x, y = ctypes.c_double(), ctypes.c_double()
            glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
            DRAG_START = vec2(x.value, y.value)
            SCALE_START = SCALE
            X_START, Y_START = X, Y
            sx, sy = (x.value-X)/SCALE, (y.value-Y)/SCALE
            DRAGGING = next((q for q in boxes if q.x<=sx<=q.x+q.w and q.y<=sy<=q.y+q.h), None)
            tol = 10 / SCALE
            RESIZING_POS = None
            if box_selected is not None:
                if box_selected.x-tol <= sx <= box_selected.x+box_selected.w+tol and box_selected.y-tol <= sy <= box_selected.y+box_selected.h+tol:
                    top, bottom = sy<=box_selected.y, sy >= box_selected.y+box_selected.h
                    if sx<=box_selected.x: # left
                        if top: RESIZING_POS = Dir.TOP_LEFT
                        elif bottom: RESIZING_POS = Dir.BOTTOM_LEFT
                        else: RESIZING_POS = Dir.LEFT
                    elif sx >= box_selected.x+box_selected.w: # right
                        if top: RESIZING_POS = Dir.TOP_RIGHT
                        elif bottom: RESIZING_POS = Dir.BOTTOM_RIGHT
                        else: RESIZING_POS = Dir.RIGHT
                    elif top: RESIZING_POS = Dir.TOP
                    elif bottom: RESIZING_POS = Dir.BOTTOM
            RESIZING = RESIZING_POS is not None
            if RESIZING is True: RESIZING_START_X, RESIZING_START_Y, RESIZING_START_W, RESIZING_START_H = (b:=box_selected).x, b.y, b.w, b.h
            else:
                if DRAGGING is None:
                    box_selected = None
                    box_selected_quad.replace([0, 0, 0, 0, 0, 0, 0, 0, 0])
                elif ui_selected is move_button and box_selected is not DRAGGING:
                    box_selected_quad.replace([DRAGGING.x, DRAGGING.y, 0, DRAGGING.w, DRAGGING.h, (c:=SELECTION_COLOR).r, c.g, c.b, 0.5])
                    box_selected = DRAGGING

            for e in ui_elements:
                if e.x<=x.value<=e.x+e.w and e.y<=y.value<=e.y+e.h: e.select()

        elif action == GLFW_RELEASE:
            DRAGGING = DRAG_START = SCALE_START = X_START = Y_START = None
            RESIZING = False
            RESIZING_POS = RESIZING_START_X = RESIZING_START_Y = RESIZING_START_W = RESIZING_START_H = None

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

cursor_arrow = glfwCreateStandardCursor(GLFW_ARROW_CURSOR) # default cursor
cursor_ns = glfwCreateStandardCursor(GLFW_RESIZE_NS_CURSOR)
cursor_ew = glfwCreateStandardCursor(GLFW_RESIZE_EW_CURSOR)
cursor_nwse = glfwCreateStandardCursor(GLFW_RESIZE_NWSE_CURSOR)
cursor_nesw = glfwCreateStandardCursor(GLFW_RESIZE_NESW_CURSOR)

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
ui_selected_quad = QuadBuffer(base_quad, quad_shader) # for ui selection
box_selected_quad = QuadBuffer(base_quad, quad_shader) # for selecting boxes

# button background:
Box(ui_quad, 0, 0, 0.2, 160, 40, 0.1, 0.1, 0.1)

move_button = Button(0, 0, 40, 40, Path(__file__).parent / "assets/images/move_tool.png")
move_button.select() # default tool
pan_button = Button(40, 0, 40, 40, Path(__file__).parent / "assets/images/pan_tool.png")
zoom_button = Button(80, 0, 40, 40, Path(__file__).parent / "assets/images/zoom_tool.png")
box_button = Button(120, 0, 40, 40, Path(__file__).parent / "assets/images/box_tool.png")

if FILEPATH.exists():
    with open(FILEPATH, "r") as f: data = json.load(f)
    for box in data["boxes"]: boxes.add(Box(quad, box["x"], box["y"], box["z"], box["w"], box["h"], box["r"], box["g"], box["b"], box["a"]))

first_setup = True
while not glfwWindowShouldClose(window):

    if QUADS_UPDATED:
        quad.clear()
        for q in boxes: q.update()
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
    box_selected_quad.draw(scale, offset)
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

glfwTerminate()
