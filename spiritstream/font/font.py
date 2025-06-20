from dataclasses import dataclass
from typing import List
from spiritstream.vec import vec2

class Font:
    def __init__(self, path:str):
        assert path.endswith(".ttf"), f"Can't load {path}. Only True Type fonts (.tff) are supported."
        from spiritstream.font import ttf
        self.engine = ttf.TTF(path)

    def render(self, char:str, fontsize:int, dpi:int, antialiasing:int=5) -> "Glyph":
        assert len(char) == 1, f"Can't render '{char}', can only render one character at a time."
        assert fontsize > 0 and dpi > 0, f"Can't render with {fontsize=} at {dpi=}. Both values must be positive."
        assert antialiasing >= 0, f"Can't render with negative antialiasing of {antialiasing}."
        return self.engine.render(ord(char), fontsize=fontsize, dpi=dpi, antialiasing=antialiasing)

@dataclass
class Glyph:
    bitmap:List[List[int]]
    size:vec2
    bearing:vec2
    advance:float

    # note: can't be frozen because is assigned texture coordinates in the glyph atlas