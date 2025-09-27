from pathlib import Path
import ctypes

spng = ctypes.CDLL(Path(__file__).resolve().parent / "build/libspng.so")

def f(name, restype, argtypes):
    func = getattr(spng, name)
    func.restype, func.argtypes = restype, argtypes
    return func

# Enums (minimal set needed for basic decoding)
class SpngErrno(ctypes.c_int):
    OK = 0

class SpngColorType(ctypes.c_int):
    GRAYSCALE = 0
    TRUECOLOR = 2
    INDEXED = 3
    GRAYSCALE_ALPHA = 4
    TRUECOLOR_ALPHA = 6

class SpngFormat(ctypes.c_int):
    RGB8 = 4
    RGBA8 = 1
    G8 = 64
    GA8 = 16

# Structs
class spng_ihdr(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("bit_depth", ctypes.c_uint8),
        ("color_type", ctypes.c_uint8),
        ("compression_method", ctypes.c_uint8),
        ("filter_method", ctypes.c_uint8),
        ("interlace_method", ctypes.c_uint8),
    ]

# Functions
spng_ctx_new = f("spng_ctx_new", ctypes.c_void_p, [ctypes.c_int])
spng_ctx_free = f("spng_ctx_free", None, [ctypes.c_void_p])
spng_set_png_buffer = f("spng_set_png_buffer", ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t])
spng_get_ihdr = f("spng_get_ihdr", ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(spng_ihdr)])
spng_decoded_image_size = f("spng_decoded_image_size", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)])
spng_decode_image = f("spng_decode_image", ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int])