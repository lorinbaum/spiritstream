from spiritstream.bindings.opengl import *
from typing import List, Dict, Tuple

# from https://github.com/tinygrad/tinygrad/blob/master/tinygrad/helpers.py
def fully_flatten(l):
  if hasattr(l, "__len__") and hasattr(l, "__getitem__") and not isinstance(l, str):
    flattened = []
    for li in l: flattened.extend(fully_flatten(li))
    return flattened
  return [l]

formats = {
   GL_RED: 1,
#    GL_RGB: 3 # TODO: add support
}
class TextureAtlas:
    def __init__(self, fmt:int, textures:Dict[str, List[float]]=None, size:int=256):
        assert fmt in formats.keys()
        self.fmt = fmt
        self.coordinates:Dict[str, Tuple[float]] = dict()
        self.channels = formats[fmt]
        self.size = size
        self.bitmap = [[0 for row in range(size)] for column in range(size)]
        # self.bitmap = [[[0 for c in range(self.channels)] for row in range(size)] for column in range(size)]
        self.skyline:List[Tuple] = [(0, 0)]
        self._texture_setup()
        if textures is not None:
            for k, v in textures.items(): self.add(k, v)

    def _texture_setup(self):
        """Exists separately to initalize the OpenGL texture after loading a TextureAtlas from .pkl"""
        self.texture = ctypes.c_uint()
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glGenTextures(1, ctypes.byref(self.texture))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        flatex = fully_flatten(self.bitmap)
        glTexImage2D(GL_TEXTURE_2D, 0, self.fmt, self.size, self.size, 0, self.fmt, GL_UNSIGNED_BYTE, (ctypes.c_ubyte * len(flatex))(*flatex))
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, 0)

    def add(self, id:str, bitmap:List[float]):
        """ Uses the Skyline algorithm for 2D packing of textures:
        Finds the lowest leftmost position to fit a rectangle of size (w, h).
        self.skyline a list of (x, y) pairs, marking the start of each "step" in the skyline profile.
        """
        if len(bitmap) == 0: return
        assert isinstance(bitmap, list) and isinstance(bitmap[0], list), "Bitmap must be 3D list: rows x column x channels. Values must be either float or int. "
        if self.channels == 1: assert isinstance(bitmap[0][0], int), bitmap[0][0]
        else: assert isinstance(bitmap[0][0], list) and isinstance(bitmap[0][0][0], int)
        assert isinstance(bitmap[0][0], int) or len(bitmap[0][0]) == self.channels, f"Mixed channels not supported. This atlas uses textures with {self.channels} channels, found {len(bitmap[0][0])} in new bitmap."

        h, w = len(bitmap), len(bitmap[0])
        pos = None
        for i, (x, y) in enumerate(self.skyline):
            if x + w > self.size: continue
            max_y, curr_x, width_left, j = y, x, w, i
            while width_left > 0 and j < len(self.skyline):
                next_x = self.skyline[j+1][0] if j+1 < len(self.skyline) else self.size
                seg = min(width_left, next_x - curr_x)
                max_y = max(max_y, self.skyline[j][1])
                width_left -= seg
                curr_x += seg
                j += 1
            if width_left > 0 or max_y + h > self.size: continue
            if pos is None or max_y < pos[2] or (max_y == pos[2] and x < pos[1]): pos = (i, x, max_y)
        if pos is None:
            from spiritstream.image import Image
            from pathlib import Path
            p = Path(__file__).parent.parent / "GlyphAtlas.bmp"
            Image.write(list(reversed(self.bitmap)), p)
            raise NotImplementedError(f"Could not fit texture {w}x{h} in the atlas. For debugging, writing bitmap to {p}")
        i, x, y = pos
        # Update skyline: insert new step and remove covered steps
        self.skyline.insert(i, (x, y + h))
        curr_x = x + w
        j = i + 1
        while j < len(self.skyline) and self.skyline[j][0] < curr_x: del self.skyline[j]
        if j == len(self.skyline) or self.skyline[j][0] > curr_x: self.skyline.insert(j, (curr_x, y))
        ret = self.coordinates[id] = (x/self.size, y/self.size, w/self.size, h/self.size)
        
        # populate self.texture with values
        for row in range(h):
            for col in range(w):
                self.bitmap[y + row][x + col] = bitmap[row][col]
        
        glBindTexture(GL_TEXTURE_2D, self.texture)
        flatex = fully_flatten(self.bitmap)
        glTexImage2D(GL_TEXTURE_2D, 0, self.fmt, self.size, self.size, 0, self.fmt, GL_UNSIGNED_BYTE, (ctypes.c_ubyte * len(flatex))(*flatex))
        glBindTexture(GL_TEXTURE_2D, 0)
        
        return ret
