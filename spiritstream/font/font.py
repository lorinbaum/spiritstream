from dataclasses import dataclass
from typing import List
from spiritstream.vec import vec2
from typing import Union
from pathlib import Path

class Font:
    def __init__(self, path:Union[str, Path]):
        assert (path := Path(path)).suffix == ".ttf", f"Can't load {path}. Only True Type fonts (.tff) are supported."
        from spiritstream.font import ttf
        self.engine = ttf.TTF(path)

    def glyph(self, char, fontsize, dpi):
        assert len(char) == 1, f"Can't render '{char}', can only render one character at a time."
        assert fontsize > 0 and dpi > 0, f"Can't render with {fontsize=} at {dpi=}. Both values must be positive."
        return self.engine.glyph(ord(char), fontsize, dpi)

    def render(self, char:str, fontsize:int, dpi:int, antialiasing:int=5) -> List[List[int]]:
        assert len(char) == 1, f"Can't render '{char}', can only render one character at a time."
        assert fontsize > 0 and dpi > 0, f"Can't render with {fontsize=} at {dpi=}. Both values must be positive."
        assert antialiasing >= 0, f"Can't render with negative antialiasing of {antialiasing}."
        return self.engine.render(ord(char), fontsize=fontsize, dpi=dpi, antialiasing=antialiasing)

@dataclass
class Glyph:
    size:vec2
    bearing:vec2
    advance:float

    # note: can't be frozen because is assigned texture coordinates in the glyph atlas