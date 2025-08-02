import contextlib, os
from typing import ClassVar

# context variable management from: https://github.com/tinygrad/tinygrad/blob/master/tinygrad/helpers.py
def getenv(key:str, default=0): return type(default)(os.getenv(key, default))

class Context(contextlib.ContextDecorator):
  def __init__(self, **kwargs): self.kwargs = kwargs
  def __enter__(self):
    self.old_context:dict[str, int] = {k:v.value for k,v in ContextVar._cache.items()}
    for k,v in self.kwargs.items(): ContextVar._cache[k].value = v
  def __exit__(self, *args):
    for k,v in self.old_context.items(): ContextVar._cache[k].value = v

class ContextVar:
  _cache: ClassVar[dict[str, "ContextVar"]] = {}
  value: int
  key: str
  def __init__(self, key, default_value):
    if key in ContextVar._cache: raise RuntimeError(f"attempt to recreate ContextVar {key}")
    ContextVar._cache[key] = self
    self.value, self.key = getenv(key, default_value), key
  def __bool__(self): return bool(self.value)
  def __ge__(self, x): return self.value >= x
  def __gt__(self, x): return self.value > x
  def __lt__(self, x): return self.value < x

SPACES = 4 # spaces per tab

PRINT_TIMINGS, CACHE_GLYPHATLAS, SAVE_GLYPHATLAS = ContextVar("PRINT_TIMINGS", 0), ContextVar("CACHE_GLYPHATLAS", 1), ContextVar("SAVE_GLYPHATLAS", 0)