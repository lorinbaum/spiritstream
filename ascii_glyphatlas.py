import struct, math
from ttf import Interpreter, vec2

I =  Interpreter(fontfile="assets/fonts/Fira_Code_v6.2/ttf/FiraCode-Regular.ttf")

# render ascii set
width = 16
# glyph_count = I.maxp.numGlyphs # all glyphs
glyph_count = 128 # all glyphs
fontsize = 10
dpi = 150
I.debug = False

glyphs = []
for i in range(glyph_count):
    I.getGlyph(i, fontsize=fontsize, dpi=dpi, antialiasing=3)
    glyphs.append({
        "bitmap": I.g.bitmap.copy(),
        "bearing": vec2(I.g.bearing.x, I.g.bearing.y), # recreate object?
        "width": len(I.g.bitmap[0]) if len(I.g.bitmap) else 0,
        "height": len(I.g.bitmap)
    })

minX, maxX = min([g["bearing"].x for g in glyphs]), max([g["bearing"].x + g["width"] for g in glyphs])
minY, maxY = min([g["bearing"].y - g["height"] for g in glyphs]), max([g["bearing"].y for g in glyphs])
tile = vec2( # big enough for every glyph
    maxX - minX,
    maxY - minY
)
origin = vec2(-minX, minY) # relative to bottom left corner. y is because in array indexing increasing y points down

bmp = [[[0,0,0] for _ in range(tile.x * width)] for _ in range(tile.y * math.ceil(glyph_count/width))]

for i, g in enumerate(glyphs):
    tile_bottom_left_corner = vec2(tile.x * (i % width), tile.y * (i // width + 1))
    bearing = vec2(g["bearing"].x, - g["bearing"].y) # y is because in array indexing increasing y points down
    glyph_top_left_corner = tile_bottom_left_corner + origin + bearing
    for y, row in enumerate(g["bitmap"]):
        bmp_y = glyph_top_left_corner.y + y
        for x, px in enumerate(row):
            bmp_x = glyph_top_left_corner.x + x
            bmp[bmp_y][bmp_x] = [int(px*255), 0, 0]

def write_bmp(filename, pixels):
    """Write a 24-bit BMP from a 2D array of RGB tuples.
    pixels: List[List[(r, g, b)]], e.g., [[(255,0,0), (0,255,0)], [(0,0,255), (255,255,255)]]
    """
    height = len(pixels)
    width = len(pixels[0]) if height > 0 else 0
    
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
        0,                # X pixels/meter
        0,                # Y pixels/meter
        0,                # Colors in palette
        0                 # Important colors
    )
    
    # Pixel data (BGR format, rows padded to 4-byte alignment)
    row_padding = (4 - (width * 3) % 4) % 4
    pixel_data = bytearray()
    
    for row in reversed(pixels):  # BMPs are stored bottom-to-top
        for r, g, b in row:
            pixel_data.extend([b, g, r])  # BMP uses BGR order
        pixel_data.extend(b'\x00' * row_padding)
    
    with open(filename, 'wb') as f:
        f.write(file_header)
        f.write(dib_header)
        f.write(pixel_data)

write_bmp("ascii.bmp", bmp)