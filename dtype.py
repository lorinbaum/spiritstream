class DType:
    def __init__(self, name:str, size:int, significant_bits:int, fractional_bits:int, signed:bool):
        assert significant_bits <= size * 8
        assert fractional_bits < significant_bits - (1 if signed else 0)
        self.size, self.name, self.significant_bits, self.fractional_bits, self.signed = size, name, significant_bits, fractional_bits, signed
    def __call__(self, v):
        assert type(v) in [bytes, int, float], f"Unsupported type for {self.name}: {type(v)}"
        if type(v) == bytes:
            if len(v) > self.size: v = v[:self.size]
            elif len(v) < self.size: v = b'\x00' * (self.size - len(v)) + v
            v = v[:self.size]
            ret = int.from_bytes(v, "big", signed=self.signed)
            ret = (-1 if ret<0 else 1) * (abs(ret) & (2**self.significant_bits - 1))
            return ret / 2**self.fractional_bits if self.fractional_bits > 0 else ret
        else:
            maxv = (2**(self.significant_bits - (1 if self.signed else 0)) - 1)  / 2**self.fractional_bits
            minv = 0 if not self.signed else -(2**(self.significant_bits - 1)) / 2**self.fractional_bits
            v = max(min(v, maxv), minv)
            if self.fractional_bits > 0: ret = (-1 if v <0 else 1) * (int(abs(v) * 2**self.fractional_bits) & (2**self.significant_bits-1)) / 2**self.fractional_bits
            else: ret = (-1 if v<0 else 1) * (int(abs(v)) & (2**self.significant_bits - 1))
            return ret
    def to_bytes(self, v): return int.to_bytes(int(self(v) * 2**self.fractional_bits), self.size, "big", signed=self.signed)

uint8 = DType("uint8", 1, 8, 0, False)
shortFrac = DType("shortFrac", 4, 32, 16, True)
FWord = DType("FWord", 2, 16, 0, True)
int16 = DType("int16", 2, 16, 0, True)
uFWord = DType("uFWord", 2, 16, 0, False)
uint16 = DType("uint16", 2, 16, 0, False)
F2Dot14 = DType("F2Dot14", 2, 16, 14, True)
Fixed = DType("Fixed", 4, 32, 16, True)
Eint8 = DType("Eint8", 4, 8, 0, True)
Euint16 = DType("Euint16", 4, 16, 0, False)
EFWord = DType("EFWord", 4, 16, 0, True)
EF2Dot14 = DType("EF2Dot14", 4, 16, 14, True)
uint32 = DType("uint32", 4, 32, 0, False)
int32 = DType("int32", 4, 32, 0, True)
F26Dot6 = DType("F26Dot6", 4, 32, 6, True)
longDateTime = DType("longDateTime", 8, 64, 0, True)