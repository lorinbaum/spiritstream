import ctypes

# opengl reference: https://registry.khronos.org/OpenGL-Refpages/gl4/
# xml: https://raw.githubusercontent.com/KhronosGroup/OpenGL-Registry/refs/heads/main/xml/gl.xml

gl = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libGL.so")
glXGetProcAddress = gl.glXGetProcAddress
glXGetProcAddress.argtypes = [ctypes.c_char_p]
glXGetProcAddress.restype = ctypes.c_void_p

def glfunc(name, restype, argtypes):
    addr = glXGetProcAddress(name.encode('utf-8'))
    if not addr: raise RuntimeError(f"Function {name} not found")
    return ctypes.CFUNCTYPE(restype, *argtypes)(addr)

# macros
GL_COLOR_BUFFER_BIT = 0x00004000
GL_ARRAY_BUFFER = 0x8892
GL_STATIC_DRAW = 0x88E4
GL_DYNAMIC_DRAW = 0x88E8
GL_VERTEX_SHADER = 0x8B31
GL_FRAGMENT_SHADER = 0x8B30
GL_COMPILE_STATUS = 0x8B81
GL_LINK_STATUS = 0x8B82
GL_FLOAT = 0x1406
GL_FALSE = 0
GL_TRIANGLES = 0x0004
GL_NO_ERROR = 0
GL_ELEMENT_ARRAY_BUFFER = 0x8893
GL_UNSIGNED_BYTE = 0x1401
GL_UNSIGNED_INT = 0x1405

GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803
GL_CLAMP_TO_EDGE = 0x812F
GL_REPEAT = 0x2901
GL_NEAREST = 0x2600
GL_LINEAR = 0x2601
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_MIN_FILTER = 0x2801
GL_RED = 0x1903
GL_RGB = 0x1907
GL_FRONT_AND_BACK = 0x0408
GL_LINE = 0x1B01
GL_TEXTURE0 = 0x84C0
GL_BLEND = 0x0BE2
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_UNPACK_ALIGNMENT = 0x0CF5

glViewport = glfunc("glViewport", None, [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int])
glClear = glfunc("glClear", None, [ctypes.c_uint])
glClearColor = glfunc("glClearColor", None, [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float])
glGenBuffers = glfunc("glGenBuffers", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
glBindBuffer = glfunc("glBindBuffer", None, [ctypes.c_uint, ctypes.c_uint])
glBufferData = glfunc("glBufferData", None, [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint])
glCreateShader = glfunc("glCreateShader", ctypes.c_uint, [ctypes.c_uint])
glShaderSource = glfunc("glShaderSource", None, [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int)])
glCompileShader = glfunc("glCompileShader", None, [ctypes.c_uint])
glGetShaderiv = glfunc("glGetShaderiv", None, [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)])
glGetShaderInfoLog = glfunc("glGetShaderInfoLog", None, [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p])
glCreateProgram = glfunc("glCreateProgram", ctypes.c_uint, [])
glAttachShader = glfunc("glAttachShader", None, [ctypes.c_uint, ctypes.c_uint])
glLinkProgram = glfunc("glLinkProgram", None, [ctypes.c_uint])
glGetProgramiv = glfunc("glGetProgramiv", None, [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)])
glGetProgramInfoLog = glfunc("glGetProgramInfoLog", None, [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.c_char_p])
glUseProgram = glfunc("glUseProgram", None, [ctypes.c_uint])
glDeleteShader = glfunc("glDeleteShader", None, [ctypes.c_uint])
glVertexAttribPointer = glfunc("glVertexAttribPointer", None, [ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_bool, ctypes.c_int, ctypes.c_void_p])
glEnableVertexAttribArray = glfunc("glEnableVertexAttribArray", None, [ctypes.c_uint])
glGenVertexArrays = glfunc("glGenVertexArrays", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
glBindVertexArray = glfunc("glBindVertexArray", None, [ctypes.c_uint])
glDrawArrays = glfunc("glDrawArrays", None, [ctypes.c_uint, ctypes.c_int, ctypes.c_int])
glDeleteVertexArrays = glfunc("glDeleteVertexArrays", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
glDeleteBuffers = glfunc("glDeleteBuffers", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
glDeleteProgram = glfunc("glDeleteProgram", None, [ctypes.c_uint])
glGetError = glfunc("glGetError", ctypes.c_uint, [])
glDrawElements = glfunc("glDrawElements", None, [ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_uint])

glGenTextures = glfunc("glGenTextures", None, [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
glBindTexture = glfunc("glBindTexture", None, [ctypes.c_uint, ctypes.c_uint])
glTexParameteri = glfunc("glTexParameteri", None, [ctypes.c_uint, ctypes.c_uint, ctypes.c_int])
glTexImage2D = glfunc("glTexImage2D", None, [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p])
glActiveTexture = glfunc("glActiveTexture", None, [ctypes.c_uint])
glGetUniformLocation = glfunc("glGetUniformLocation", ctypes.c_int, [ctypes.c_uint, ctypes.c_char_p])
glUniform1i = glfunc("glUniform1i", None, [ctypes.c_int, ctypes.c_int])
glUniform3f = glfunc("glUniform3f", None, [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float])
glEnable = glfunc("glEnable", None, [ctypes.c_uint])
glBlendFunc = glfunc("glBlendFunc", None, [ctypes.c_uint, ctypes.c_uint])
glPixelStorei = glfunc("glPixelStorei", None, [ctypes.c_uint, ctypes.c_int])

glBufferSubData = glfunc("glBufferSubData", None, [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])

glPolygonMode = glfunc("glPolygonMode", None, [ctypes.c_uint, ctypes.c_uint])