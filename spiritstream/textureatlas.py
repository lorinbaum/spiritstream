from spiritstream.font import Font
from spiritstream.vec import vec2
from spiritstream.bindings.opengl import *
import math

class TextureAtlas:
    def __init__(self, chars:set, F:Font, size:int, dpi:int):
        self.glyphs = {c: F.render(c, size, dpi) for c in chars}

        grid_size = math.ceil(math.sqrt(len(chars)))
        xMin, xMax = min(g.size.x for g in self.glyphs.values()), max(g.size.x for g in self.glyphs.values())
        yMin, yMax = min(g.size.y for g in self.glyphs.values()), max(g.size.y for g in self.glyphs.values())

        self.tile = vec2(xMax - xMin, yMax - yMin) # big enough for every glyph

        texture_size = self.tile * grid_size
        bmp = [[0 for x in range(texture_size.x)] for row in range(texture_size.y)] # single channel
        self.origin = vec2(-xMin, -yMin)
        for i, v in enumerate(self.glyphs.values()):
            v.texture_coords = [
                (top_left:=vec2(self.tile.x * (i % grid_size), self.tile.y * (i // grid_size) + v.size.y) + self.origin), # top left
                top_left + vec2(v.size.x, 0), # top right
                top_left + vec2(0,        -v.size.y), # bottom left
                top_left + vec2(v.size.x, -v.size.y) # bottom right
            ]
            for y, row in enumerate(reversed(v.bitmap)): # bitmap starts with top row, but the new array is written assuming the first row is the bottom row. OpenGL expects textures that way.
                assert 0 <= (y_idx := v.texture_coords[2].y + y) < texture_size.y, y_idx
                for x, px in enumerate(row):
                    assert 0 <= (x_idx := v.texture_coords[2].x + x) < texture_size.x
                    bmp[y_idx][x_idx] = int(px*255) # red channel only
            v.texture_coords = [coord / texture_size for coord in v.texture_coords] # scale texture coordinates for use in vertices
        for v in self.glyphs.values(): del v.bitmap

        # texture
        self.texture = ctypes.c_uint()
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glGenTextures(1, ctypes.byref(self.texture))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        flat_bmp = [x for y in bmp for x in y]
        assert len(flat_bmp) == texture_size.x * texture_size.y
        bmp_ctypes = (ctypes.c_ubyte * len(flat_bmp))(*flat_bmp)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, texture_size.x, texture_size.y, 0, GL_RED, GL_UNSIGNED_BYTE, bmp_ctypes)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, 0)

        # from ascii_glyphatlas import write_bmp
        # write_bmp("glyph atlas.bmp", list(reversed(bmp))) 

    def __getitem__(self, i): return self.glyphs[i]