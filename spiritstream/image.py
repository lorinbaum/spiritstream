from pathlib import Path
from typing import Union
import struct

class Image:
    def read(filename:Union[Path, str]): raise NotImplementedError
    def write(data, filename):
        if (filename := Path(filename)).suffix == ".bmp": bmp_write(data, filename)
        else: raise NotImplementedError(f"{filename} has unsupported format: {filename.suffix}")

def bmp_write(data, filename):
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
    
    with open(filename, 'wb') as f:
        f.write(file_header)
        f.write(dib_header)
        f.write(pixel_data)