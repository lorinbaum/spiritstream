import ctypes

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


glClear = glfunc("glClear", None, [ctypes.c_uint])
glClearColor = glfunc("glClearColor", None, [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float])
