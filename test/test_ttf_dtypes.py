import unittest
from spiritstream.font.ttf import *

class TestDTypes(unittest.TestCase):
    def test_dtypes(self):
        dtypes = [uint8, shortFrac, FWord, int16, uFWord, uint16, F2Dot14, Fixed, Eint8, Euint16, EFWord, EF2Dot14, uint32, int32, F26Dot6, longDateTime]
        values = [0, 123, 1, 812931, -13321, 123.123123819]
        for dt in dtypes:
            for v in values:
                self.assertEqual(dt(v), dt(dt.to_bytes(v)))
                if 0 < (v1:=int(v)) < 256 and dt not in [EF2Dot14, F2Dot14]: self.assertEqual(v1, dt(v1))

if __name__ == "__main__": unittest.main()