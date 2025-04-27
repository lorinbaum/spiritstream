import math
# Instruction helpers
class vec2:
    def __init__(self, x=None, y=None):
        self.types = [int, float, vec2] # supported types for arithmetic
        self.x = x
        self.y = y
    def mag(self): return math.sqrt(self.x**2 + self.y**2)
    def normalize(self): return vec2(self.x / self.mag(), self.y / self.mag())
    def components(self): return [self.x, self.y]
    def copy(self): return vec2(self.x, self.y)
    def dot(self, other) -> int:
        assert type(other) == vec2
        return sum([self.x * other.x, self.y * other.y])
    def __add__(self, other):
        assert type(other) in self.types
        return vec2(self.x + other.x, self.y + other.y) if type(other) == vec2 else vec2(self.x + other, self.y + other)
    def __sub__(self, other):
        assert type(other) in self.types
        return vec2(self.x - other.x, self.y - other.y) if type(other) == vec2 else vec2(self.x - other, self.y - other)
    def __rsub__(self, other):
        assert type(other) in self.types
        return vec2(other.x - self.x, other.y - self.y) if type(self) == vec2 else vec2(other - self, other - self)
    def __mul__(self, other):
        assert type(other) in self.types
        return vec2(self.x * other.x, self.y * other.y) if type(other) == vec2 else vec2(self.x * other, self.y * other)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other):
        assert type(other) in self.types
        return vec2(self.x / other.x, self.y / other.y) if type(other) == vec2 else vec2(self.x / other, self.y / other)
    def __rtruediv__(self, other):
        assert type(other) in self.types
        return vec2(other.x / self.x, other.y / self.y) if type(other) == vec2 else vec2(other / self.x, other / self.y)
    def __repr__(self): return f"vec2({self.x}, {self.y})"
    def __eq__(self, other): return self.x == other.x and self.y == other.y