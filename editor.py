from glfw import *
from opengl import *

glfwInit()
glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2)
glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, True)
glfwWindowHint(GLFW_RESIZABLE, False)

window = glfwCreateWindow(800, 600, b"Spiritstream", None, None)
glfwMakeContextCurrent(window)

while not glfwWindowShouldClose(window):
    if glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS: glfwSetWindowShouldClose(window, GLFW_TRUE)
    glClearColor(0.3, 0.2, 0.2, 1)
    glClear(GL_COLOR_BUFFER_BIT)
    glfwSwapBuffers(window)
    glfwPollEvents()

glfwTerminate()