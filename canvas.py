"""
3 squares that I can move around
"""
from spiritstream.bindings.opengl import *
from spiritstream.bindings.glfw import *
from spiritstream.shader import Shader
from spiritstream.buffer import BaseQuad, QuadBuffer
from spiritstream.vec import vec2

WIDTH = 1000
HEIGHT = 1000
RESIZED = True
X = 0
Y = 0
quads = set()
DRAGGING = None
DRAGGING_OFFSET = None
QUADS_UPDATED = True

class Quad:
    def __init__(self, x, y, z, width, height, red, green, blue, alpha=1.0):
        self.x, self.y, self.z, self.w, self.h, self.r, self.g, self.b, self.a = x, y, z, width, height, red, green, blue, alpha
        self.update()
    def __setattr__(self, name, value):
        global QUADS_UPDATED
        QUADS_UPDATED = True
        super().__setattr__(name, value)
    def update(self):
        quad.add([self.x, self.y, self.z, self.w, self.h, self.r, self.g, self.b, self.a])
    def __repr__(self): return f"Quad(x={self.x}, y={self.y}, w={self.w}, h={self.h}, RGBA=({self.r},{self.g},{self.b},{self.a}))"

@GLFWframebuffersizefun
def framebuffer_size_callback(window, width, height):
    global WIDTH, HEIGHT, RESIZED
    WIDTH, HEIGHT, RESIZED = width, height, True

@GLFWkeyfun
def key_callback(window, key:int, scancode:int, action:int, mods:int):
    if action in [GLFW_PRESS, GLFW_REPEAT]:
        if key == GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GLFW_TRUE)

@GLFWcursorposfun
def cursor_pos_callback(window, x, y):
    global DRAGGING, DRAGGING_OFFSET
    if glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS:
        x, y = ctypes.c_double(), ctypes.c_double()
        glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
        if DRAGGING is not None is not DRAGGING_OFFSET:
            DRAGGING.x = x.value - DRAGGING_OFFSET.x
            DRAGGING.y = y.value - DRAGGING_OFFSET.y

@GLFWmousebuttonfun
def mouse_callback(window, button:int, action:int, mods:int):
    if button == GLFW_MOUSE_BUTTON_1:
        global DRAGGING, DRAGGING_OFFSET
        if action is GLFW_PRESS:
            x, y = ctypes.c_double(), ctypes.c_double()
            glfwGetCursorPos(window, ctypes.byref(x), ctypes.byref(y))
            x, y = x.value, y.value
            DRAGGING = next((q for q in quads if q.x<=x<=q.x+q.w and q.y<=y<=q.y+q.h), None)
            if DRAGGING is not None: DRAGGING_OFFSET = vec2(x-DRAGGING.x, y-DRAGGING.y)
        elif action == GLFW_RELEASE: DRAGGIN = DRAGGING_OFFSET = None

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
glEnable(GL_DEPTH_TEST)
glClearDepth(1)
glClearColor(0, 0, 0, 1)

quad_shader = Shader(p:=Path(__file__).parent / "spiritstream/shaders/quad.vert", p.parent / "quad.frag", ["scale", "offset"])

base_quad = BaseQuad()
quad = QuadBuffer(base_quad, quad_shader)

quads.add(Quad(100, 300, 0, 100, 100, 0.5, 0.5, 0)) #  yellow
quads.add(Quad(120, 320, 0, 100, 100, 0.5, 0.5, 1.0)) # blue
quads.add(Quad(140, 340, 0, 100, 100, 0.5, 0, 0.5)) # magenta

while not glfwWindowShouldClose(window):

    if QUADS_UPDATED:
        quad.clear()
        for q in quads: q.update()
        QUADS_UPDATED = False

    if RESIZED:
        glViewport(0, 0, WIDTH, HEIGHT)
        RESIZED = False

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    scale, offset = (2 / WIDTH, -2 / HEIGHT), tuple(vec2(-1 + X * 2 / WIDTH, 1 + Y * -2 / HEIGHT).components())
    quad.draw(scale, offset)

    glfwSwapBuffers(window)
    glfwPollEvents()

base_quad.delete()
quad_shader.delete()
quad.delete()
glfwTerminate()
