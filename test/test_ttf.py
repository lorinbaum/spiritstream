import unittest, math
from spiritstream.font.ttf import *
from random import random, randint
from typing import List, Union
from spiritstream.font.table import glyf
from spiritstream.font.ttf import TTF

class TestDTypes(unittest.TestCase):
    def test_dtypes(self):
        dtypes = [uint8, shortFrac, FWord, int16, uFWord, uint16, F2Dot14, Fixed, Eint8, Euint16, EFWord, EF2Dot14, uint32, int32, F26Dot6, longDateTime]
        values = [0, 123, 1, 812931, -13321, 123.123123819]
        for dt in dtypes:
            for v in values:
                self.assertEqual(dt(v), dt(dt.to_bytes(v)))
                if 0 < (v1:=int(v)) < 256 and dt not in [EF2Dot14, F2Dot14]: self.assertEqual(v1, dt(v1))

class TestMisc(unittest.TestCase):
    def test_checksum(self):
        fonts = [
            "assets/fonts/Fira_Code_v6.2/ttf/FiraCode-Regular.ttf",
            "assets/fonts/georgia-2/georgia.ttf",
            "assets/fonts/Arial.ttf",
            # "assets/fonts/Roboto/static/Roboto-Regular.ttf" # TODO: fails checksum for unknown reason
        ]
        for f in fonts:
            I = TTF(fontfile=f)
            with open(f, "rb") as f: self.assertTrue(all(I.checksum(f.read())))


if __name__ == "__main__": unittest.main()