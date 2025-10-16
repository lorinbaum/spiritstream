import ctypes
from pathlib import Path
from enum import Enum

nanojpg = ctypes.CDLL(Path(__file__).resolve().parent / "build/libnanojpeg.so")

def f(name, restype, argtypes):
    func = getattr(nanojpg, name)
    func.restype, func.argtypes = restype, argtypes
    return func

class NJResult(Enum):
    OK = 0
    NO_JPEG = 1
    UNSUPPORTED = 2
    OUT_OF_MEM = 3
    INTERNAL_ERR = 4
    SYNTAX_ERROR = 5

njInit = f("njInit", None, [])
njDecode = f("njDecode", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int])
njGetWidth = f("njGetWidth", ctypes.c_int, [])
njGetHeight = f("njGetHeight", ctypes.c_int, [])
njIsColor = f("njIsColor", ctypes.c_int, [])
njGetImage = f("njGetImage", ctypes.POINTER(ctypes.c_ubyte), [])
njGetImageSize = f("njGetImageSize", ctypes.c_int, [])
njDone = f("njDone", None, [])