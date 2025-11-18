from pathlib import Path
import struct, ctypes
from typing import List
from dataclasses import dataclass

@dataclass(frozen=True)
class Image:
    format:str
    width:int
    height:int
    data:List
    color:bool
    alpha:bool

def read(path:Path) -> Image:
    match path.suffix:
        case ".jpg": return jpg_read(path)
        case ".png": return png_read(path)
        case _: raise NotImplementedError(f"Writing {path.suffix} format not supported")

def write(data, path:Path):
    match path.suffix:
        case ".bmp": bmp_write(data, path)
        case _: raise NotImplementedError(f"Reading {path.suffix} format not supported")
        

def bmp_write(data, path):
    """Write a 24-bit BMP from a 2D array of RGB tuples.
    data: List[List[(r, g, b)]], e.g., [[(255,0,0), (0,255,0)], [(0,0,255), (255,255,255)]]
    """
    height = len(data)
    width = len(data[0]) if height > 0 else 0
    
    # BMP headers
    file_header = struct.pack(
        '<2sIHHI',
        b'BM',            # Magic
        54 + 3*width*height,  # File size
        0,                # Reserved
        0,                # Reserved
        54                # Pixel data offset
    )
    
    dib_header = struct.pack(
        '<IIIHHIIIIII',
        40,               # Header size
        width,            # Width
        height,           # Height
        1,                # Planes
        24,               # Bits per pixel
        0,                # Compression (BI_RGB)
        3*width*height,   # Image size
        0,                # X data/meter
        0,                # Y data/meter
        0,                # Colors in palette
        0                 # Important colors
    )
    
    # Pixel data (BGR format, rows padded to 4-byte alignment)
    row_padding = (4 - (width * 3) % 4) % 4
    pixel_data = bytearray()
    
    for row in reversed(data):  # BMPs are stored bottom-to-top
        # if len(row) == 3:
        #     for r, g, b in row:
        #         pixel_data.extend([b, g, r])  # BMP uses BGR order
        #     pixel_data.extend(b'\x00' * row_padding)
        # else:
        for v in row: pixel_data.extend([v, v, v])
        pixel_data.extend(b'\x00' * row_padding)
    
    with open(path, 'wb') as f:
        f.write(file_header)
        f.write(dib_header)
        f.write(pixel_data)

def jpg_read(path):
    from spiritstream.bindings.nanojpeg import njInit, njDecode, njDone, njGetWidth, njGetHeight, njIsColor, njGetImage, njGetImageSize, NJResult
    with open(path, "rb") as f: jpg = f.read()
    buf = ctypes.create_string_buffer(jpg)

    njInit()
    res = NJResult(njDecode(ctypes.cast(buf, ctypes.c_void_p), len(jpg)))

    if res is not NJResult.OK:
        njDone()
        raise RuntimeError(res.name)
    else:
        width = njGetWidth()
        height = njGetHeight()
        is_color = njIsColor() != 0
        image_size = njGetImageSize()
        image_ptr = njGetImage()  # Pointer to raw image data (unsigned char*)
        # image_data = ctypes.cast(image_ptr, ctypes.POINTER(ctypes.c_ubyte * image_size)).contents # for python data
        image_copy = (ctypes.c_ubyte * image_size)()
        ctypes.memmove(image_copy, image_ptr, image_size)
        njDone()
        return Image("jpg", width, height, image_copy, is_color, False)
    
def png_read(path):
    from spiritstream.bindings.spng import spng_ctx_new, spng_set_png_buffer, SpngErrno, spng_ctx_free, spng_ihdr, spng_get_ihdr, SpngColorType, SpngFormat, \
        spng_decoded_image_size, spng_decode_image
    with open(path, "rb") as f: png = f.read()
    buf = ctypes.create_string_buffer(png)

    ctx = spng_ctx_new(0)
    if not ctx: raise RuntimeError("Failed to create SPNG context")

    res = spng_set_png_buffer(ctx, ctypes.cast(buf, ctypes.c_void_p), len(png))
    if res is not SpngErrno.OK:
        spng_ctx_free(ctx)
        raise RuntimeError("Failed to set PNG buffer")

    ihdr = spng_ihdr()
    res = spng_get_ihdr(ctx, ctypes.byref(ihdr))
    if res is not SpngErrno.OK:
        spng_ctx_free(ctx)
        raise RuntimeError("Failed to get IHDR")

    width = ihdr.width
    height = ihdr.height
    color_type = ihdr.color_type


    is_color = color_type in (SpngColorType.TRUECOLOR, SpngColorType.TRUECOLOR_ALPHA, SpngColorType.INDEXED)
    alpha = color_type is SpngColorType.TRUECOLOR_ALPHA # strip alpha on grayscale alpha images
    # Not supporting grayscale color
    fmt = SpngFormat.G8 if not is_color else SpngFormat.RGBA8 if alpha else SpngFormat.RGB

    image_size = ctypes.c_size_t()
    res = spng_decoded_image_size(ctx, fmt, ctypes.byref(image_size))
    if res is not SpngErrno.OK:
        spng_ctx_free(ctx)
        raise RuntimeError("Failed to get decoded image size")

    image_copy = (ctypes.c_ubyte * image_size.value)()
    res = spng_decode_image(ctx, image_copy, image_size.value, fmt, 0)
    if res is not SpngErrno.OK:
        spng_ctx_free(ctx)
        raise RuntimeError("Failed to decode image")

    spng_ctx_free(ctx)
    return Image("png", width, height, image_copy, is_color, alpha)