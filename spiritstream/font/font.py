from spiritstream.font import ttf
from dataclasses import dataclass
from typing import List
from spiritstream.vec import vec2

class Font:
    def __init__(self, path:str):
        assert path.endswith(".ttf"), f"Can't load {path}. Only True Type fonts (.tff) are supported."
        self.interpreter = ttf.Interpreter(path)

    def render(self, char:str, fontsize:int, dpi:int, antialiasing:int=3) -> "Glyph":
        assert len(char) == 1, f"Can't render '{char}', can only render one character at a time."
        assert fontsize > 0 and dpi > 0, f"Can't render with {fontsize=} at {dpi=}. Both values must be positive."
        assert antialiasing >= 0, f"Can't render with negative antialiasing of {antialiasing}."
        g = self.interpreter.getGlyph(ord(char), fontsize=fontsize, dpi=dpi, antialiasing=antialiasing)
        size = vec2(len(g.bitmap[0]) if len(g.bitmap) else 0, len(g.bitmap))
        return Glyph(g.bitmap.copy(), size, g.bearing.copy(), g.advanceWidth)

@dataclass
class Glyph:
    bitmap:List[List[int]]
    size:vec2
    bearing:vec2
    advance:int

