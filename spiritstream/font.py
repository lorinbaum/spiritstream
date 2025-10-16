from dataclasses import dataclass
from typing import List, Tuple
from spiritstream.vec import vec2
from spiritstream.dtype import *
from spiritstream.helpers import quadratic_equation
from typing import Union
from pathlib import Path
import math, functools

"""
- Parser / Serializer for dtypes and TrueType tables
- TrueType table classes for parsing
- TrueType engine to load fonts, tables and glyphs and canonicalize the contours to be rasterized by the
- general Curve rasterizer (supports Super-Sampled Anti-Aliasing)
- higher level font interface to get glyph objects (size, bearing, advance) or bitmaps
"""

# PARSER / SERIALIZER FOR DTYPES AND TABLES

class Parser:
    """Conveniently parse dtypes or tables with automatic pointer incrementing"""
    def __init__(self, buffer:bytes):
        self.b = buffer
        self.p = 0 # pointer
    def parse(self, dt:Union[DType, "table"], count:int=1, offset:int=None, **kwargs) -> Union[int, List]:
        assert count >= 0, f"Got invalid count {count}"
        if count == 0: return []
        ret = []
        if offset: self.p = offset
        for i in range(count):
            if hasattr(dt, "size"): # dtypes have size, tables don't
                assert len(self.b) >= self.p + dt.size
                ret.append(dt(self.b[self.p : self.p+dt.size]))
                self.p += dt.size
            else: ret.append(dt(self, **kwargs)) # self.p will still increment because all tables are made from dtypes and they have sizes. kwargs are needed for hmtx table
        return ret if len(ret) > 1 else ret[0]

class Serializer:
    """Given a buffer of bytes, it's method serialize can be called with a dtype and a value and it will serialize it into the buffer.
    It exists as a wrapper around the buffer to automatically increment the pointer"""
    def __init__(self, size:int):
        self.size = size
        self.b = b''
    def serialize(self, dt:DType, v:Union[int, float]): # works only with dtypes, not tables
        self.b += dt.to_bytes(v)
        assert len(self.b) <= self.size, f"Error: Exceeded expected buffer size. {self.size=}"

# TRUE TYPE TABLES

class table:
    def __init__(self, b:Union[bytes, Parser], **kwargs):
        assert type(b) in [bytes, Parser]
        if type(b) == bytes: b = Parser(b)
        self._from_bytes(b, **kwargs)
    def _from_bytes(self): raise NotImplementedError
    def to_bytes(self):
        self.s = Serializer(sum([dt.size for dt in self.types.values()]))
        for dt, (_, v) in zip(self.types.values(), self.__dict__.items()): self.s.serialize(dt, v)
        return self.s.b # only works for those simple tables that have the "types" class variable 

class offset_subtable_table(table):
    types = {
        "scaler_type": uint32,
        "numTables": uint16,
        "searchRange": uint16,
        "entrySelector": uint16,
        "rangeShift": uint16
    }
    def _from_bytes(self, b:Parser): [setattr(self, name, b.parse(dt)) for name, dt in self.types.items()]

class table_directory_entry(table):
    types = {
        "tag": uint32,
        "checkSum": uint32,
        "offset": uint32,
        "length": uint32,
    }
    def _from_bytes(self, b:Parser): [setattr(self, name, b.parse(dt)) for name, dt in self.types.items()]

class head_table(table):
    types = {
        "version": Fixed,
        "fontRevision": Fixed,
        "checkSumAdjustment": uint32,
        "magicNumber": uint32,
        "flags": uint16,
        "unitsPerEM": uint16,
        "created": longDateTime,
        "modified": longDateTime,
        "xMin": FWord,
        "yMin": FWord,
        "xMax": FWord,
        "yMax": FWord,
        "macStyle": uint16,
        "lowestRecPPEM": uint16,
        "fontDirectionHint": uint16,
        "indexToLocFormat": uint16,
        "glyphDataFormat": uint16
    }
    def _from_bytes(self, b:Parser): [setattr(self, name, b.parse(dt)) for name, dt in self.types.items()]

class maxp_table(table):
    def _from_bytes(self, b:Parser):
        self.version = b.parse(Fixed)
        self.numGlyphs = b.parse(uint16)
        self.maxPoints = b.parse(uint16)
        self.maxContours = b.parse(uint16)
        self.maxComponentPoints = b.parse(uint16)
        self.maxComponentContours = b.parse(uint16)
        self.maxZones = b.parse(uint16)
        self.maxTwilightPoints = b.parse(uint16)
        self.maxStorage = b.parse(uint16)
        self.maxFunctionDefs = b.parse(uint16)
        self.maxInstrucionDefs = b.parse(uint16)
        self.maxStackElements = b.parse(uint16)
        self.maxSizeOfInstructions = b.parse(uint16)
        self.maxComponentElements = b.parse(uint16)
        self.maxComponentDepth = b.parse(uint16)

class cmap_table(table):
    def _from_bytes(self, b:Parser, length=None):
        assert length != None
        table_offset = b.p
        self.version = b.parse(uint16)
        self.numberSubtables = b.parse(uint16)
        self.encoding_subtables = b.parse(cmap_encoding_subtable, count=self.numberSubtables)
        # choose subtable
        supported = [
            [0,4], # unicode platform with unicode 2.0 and onwards semantics
            [3,1] # windows platform with unicode BMP encoding
        ]
        for ect in self.encoding_subtables:
            if [ect.platformID, ect.platformSpecificID] in supported:
                self.subtable = b.parse(cmap_subtable, offset=table_offset+ect.offset, table_offset=table_offset, length=length)
                break
        assert hasattr(self, "subtable"), f"No supported subtable found. Available tables: {[(ect.platformID, ect.platformSpecificID) for ect in self.encoding_subtables]}"
    
class cmap_encoding_subtable(table):
    def _from_bytes(self, b:Parser):
        self.platformID = b.parse(uint16)
        self.platformSpecificID = b.parse(uint16)
        self.offset = b.parse(uint32)

class cmap_subtable(table):
    def _from_bytes(self, b:Parser, table_offset=None, length=None):
        self.format = b.parse(uint16)
        match self.format:
            case 4:
                assert length != None, "for cmap subtable format 4, length is required"
                self.length = b.parse(uint16)
                self.language = b.parse(uint16)
                self.segCountX2 = b.parse(uint16)
                self.segCount = self.segCountX2 // 2
                self.searchRange = b.parse(uint16)
                assert self.searchRange == ((2**math.floor(math.log2(self.segCount))) * 2)
                self.entrySelector = b.parse(uint16)
                assert self.entrySelector == math.floor(math.log2(self.segCount))
                self.rangeShift = b.parse(uint16)
                assert self.rangeShift == self.segCountX2 - self.searchRange
                self.endCode = b.parse(uint16, count=self.segCount)
                assert self.endCode[-1] == 0xFFFF
                self.reservePad = b.parse(uint16)
                assert self.reservePad == 0
                self.startCode = b.parse(uint16, count=self.segCount)
                self.idDelta = b.parse(int16, self.segCount)
                self.idRangeOffset = b.parse(uint16, count=self.segCount)
                self.glyphIdArray = b.parse(uint16, count=(table_offset + length - b.p)//2) # rest of the table is this stuff

                def getGlyphIndex(unicode:int):
                    for i in range(self.segCount):
                        if self.startCode[i] <= unicode <= self.endCode[i]:
                            if self.idRangeOffset[i] == 0: return (unicode + self.idDelta[i]) % 65536
                            else:
                                offset = self.idRangeOffset[i] // 2 + (unicode - self.startCode[i])
                                glyph_index = self.glyphIdArray[offset - (self.segCount - i)]
                                return (glyph_index + self.idDelta[i]) % 65536 if glyph_index != 0 else 0
                    return 0 # missing character glyph
                self.getGlyphIndex = getGlyphIndex

            case 12:
                self.reserved = b.parse(uint16)
                self.length = b.parse(uint32)
                self.language = b.parse(uint32)
                self.nGroups = b.parse(uint32)
                self.groups = b.parse(cmap_subtable12_group, count=self.nGroups)
                
                def getGlyphIndex(unicode:int):
                    for g in self.groups:
                        if g.startCharCode <= unicode and g.endCharCode >= unicode:
                            return g.startGlyphCode + unicode - g.startCharCode
                    return 0 # missing characer glyph
                self.getGlyphIndex = getGlyphIndex
            case _:
                print(self.format)
                raise NotImplementedError

class cmap_subtable12_group(table):
    def _from_bytes(self, b:Parser):
        self.startCharCode = b.parse(uint32)
        self.endCharCode = b.parse(uint32)
        self.startGlyphCode = b.parse(uint32)

class hhea_table(table):
    def _from_bytes(self, b:Parser):
        self.version = b.parse(Fixed)
        self.ascent = b.parse(FWord)
        self.descent = b.parse(FWord)
        self.lineGap = b.parse(FWord)
        self.advanceWidthMax = b.parse(uFWord)
        self.minLeftSideBearing = b.parse(FWord)
        self.minRightSideBearing = b.parse(FWord)
        self.xMaxExtent = b.parse(FWord)
        self.caretSlopeRise = b.parse(int16)
        self.caretSlopeRun = b.parse(int16)
        self.caretOffset = b.parse(FWord)
        self._ = b.parse(int16, count=4)
        self.metricDataFormat = b.parse(int16)
        self.numOfLongHorMetrics = b.parse(uint16)

class hmtx_table(table):
    def _from_bytes(self, b:Parser, numOfLongHorMetrics=None, numGlyphs=None):
        assert numOfLongHorMetrics != None and numGlyphs != None
        self.longHorMetric = b.parse(longHorMetric, count=numOfLongHorMetrics)
        self.leftSideBearing = b.parse(FWord, count=numGlyphs - numOfLongHorMetrics)

class longHorMetric(table):
    def _from_bytes(self, b:Parser):
        self.advanceWidth = b.parse(uint16)
        self.leftSideBearing = b.parse(int16)

class glyf(table):
    def _from_bytes(self, b:Parser):
        if len(b.b) == 0: # support glyph without outlines, like spaces
            self.numberOfContours = 0
            self.endPtsContours = []
            self.xMin = 0
            self.yMin = 0
            self.xMax = 0
            self.yMax = 0
            self.x = []
            self.y = []
            self.flags = []
        else:
            self.numberOfContours = b.parse(int16)
            self.xMin = b.parse(int16)
            self.yMin = b.parse(int16)
            self.xMax = b.parse(int16)
            self.yMax = b.parse(int16)
            if self.numberOfContours > 0:
                self.endPtsContours = b.parse(uint16, count=self.numberOfContours)
                if type(self.endPtsContours) != list: self.endPtsContours = [self.endPtsContours]
                self.instructionLength = b.parse(uint16)
                self.instructions = b.parse(uint8, count=self.instructionLength)
                self.flags = [None] * (self.endPtsContours[-1] + 1)
                i = 0
                while i < len(self.flags):
                    count = 1
                    flag = b.parse(uint8)
                    assert flag & 0xC0 == 0, "last two flags must be 0"
                    if flag & 0x8: count += b.parse(uint8)
                    for j in range(count):
                        self.flags[i] = flag
                        i += 1
                assert self.flags[-1] != None

                i = prevX = 0
                self.x = [None] * (self.endPtsContours[-1] + 1)
                while i < len(self.x):
                    x_short = self.flags[i] & 0x2
                    x_same_or_positive = self.flags[i] & 0x10
                    if x_short and x_same_or_positive: x = prevX + b.parse(uint8)
                    elif x_short and not x_same_or_positive: x = prevX - b.parse(uint8)
                    elif not x_short and x_same_or_positive: x = prevX
                    else: x = prevX + b.parse(int16)
                    self.x[i] = prevX = x
                    i += 1

                i = prevY = 0
                self.y = [None] * (self.endPtsContours[-1] + 1)
                while i < len(self.y):
                    y_short = self.flags[i] & 0x4
                    y_same_or_positive = self.flags[i] & 0x20
                    if y_short and y_same_or_positive: y = prevY + b.parse(uint8)
                    elif y_short and not y_same_or_positive: y = prevY - b.parse(uint8)
                    elif not y_short and y_same_or_positive: y = prevY
                    else: y = prevY + b.parse(int16)
                    self.y[i] = prevY = y
                    i += 1
            elif self.numberOfContours < 0:
                """
                Compound Glyph
                This glyph is made from multiple other glyphs. Instead of loading x,y and flags:
                Populate self.children with glyphComponents and loads any instructions
                """
                self.children = [glyphComponent(b)]
                while self.children[-1].MORE_COMPONENTS: self.children.append(glyphComponent(b))
                if any([child.WE_HAVE_INSTRUCTIONS for child in self.children]):
                    self.instructionLength = b.parse(uint16)
                    self.instructions = b.parse(uint8, count=self.instructionLength)

class glyphComponent(table):
    masks = {
        "ARG_1_AND_2_ARE_WORDS": 0x0001,
        "ARGS_ARE_XY_VALUES": 0x0002,
        "ROUND_XY_TO_GRID": 0x0004,
        "WE_HAVE_A_SCALE": 0x0008,
        "MORE_COMPONENTS": 0x0020,
        "WE_HAVE_AN_X_AND_Y_SCALE": 0x0040,
        "WE_HAVE_A_TWO_BY_TWO": 0x0080,
        "WE_HAVE_INSTRUCTIONS": 0x0100,
        "USE_MY_METRICS": 0x0200,
        "OVERLAP_COMPOUND": 0x0400,
        "SCALED_COMPONENT_OFFSET": 0x0800,
        "UNSCALED_COMPONENT_OFFSET": 0x1000,
        "Reserved": 0xE010
    }
    def _from_bytes(self, b:Parser):
        self.flags = b.parse(uint16)
        self.glyphIndex = b.parse(uint16)
        [setattr(self, name, bool(self.flags & value)) for name, value in glyphComponent.masks.items()] # replace class variables with their bools
        if self.ARG_1_AND_2_ARE_WORDS:
            if self.ARGS_ARE_XY_VALUES: self.arg1, self.arg2 = b.parse(int16, count=2)
            else: self.arg1, self.arg2 = b.parse(uint16, count=2)
        else:
            if self.ARGS_ARE_XY_VALUES: self.arg1, self.arg2 = b.parse(int8, count=2)
            else: self.arg1, self.arg2 = b.parse(uint8, count=2)
        if self.WE_HAVE_A_SCALE: self.scale = b.parse(F2Dot14)
        elif self.WE_HAVE_AN_X_AND_Y_SCALE: self.xscale, self.yscale = b.parse(F2Dot14, count=2)
        elif self.WE_HAVE_A_TWO_BY_TWO: self.xscale, self.scale01, self.scale10, self.yscale = b.parse(F2Dot14, count=4)
        assert self.Reserved == False # reserved bits should be set to 0
        assert int(self.WE_HAVE_A_SCALE) + int(self.WE_HAVE_AN_X_AND_Y_SCALE) + int(self.WE_HAVE_A_TWO_BY_TWO) in [0,1] # at most one of these flags can be True at a time

class post_table(table):
    def _from_bytes(self, b:Parser):
        self.format = b.parse(Fixed)
        self.italicAngle = b.parse(Fixed)
        self.underlinePosition = b.parse(FWord)
        self.underlineThickness = b.parse(FWord)
        # there is more here, but thats not needed

# TRUE TYPE ENGINE

# NOTE: can't be frozen because is assigned texture coordinates in the glyph atlas
@dataclass
class Glyph:
    size:vec2
    bearing:vec2
    advance:float

@dataclass(frozen=True)
class CurvePoint:
    x:float
    y:float
    onCurve:bool

# CHECKSUM
def pad4(b:bytes) -> bytes: return b.ljust((len(b) + 3) // 4 * 4, b'\0')
def checkSum(b:bytes) -> int: return sum([int.from_bytes(b[i:i+4], "big") for i in range(0, len(b), 4) if (b:=pad4(b))]) & 0xFFFFFFFF

class TTF:
    def __init__(self, fontfile, check=False): self.load(fontfile, check=check)
    
    def load(self, fontfile, check=False):
        with open(fontfile, "rb") as f: p = Parser(f.read())

        self.offset_subtable = p.parse(offset_subtable_table)
        self.table_directory = {uint32.to_bytes(entry.tag).decode():entry for entry in p.parse(table_directory_entry, count=self.offset_subtable.numTables)}
        if check:
            results = {}
            head_bytes = p.b[(t:=self.table_directory["head"]).offset:t.offset+t.length]
            head = head_table(head_bytes)
            results["from and to bytes"] = head_bytes == head.to_bytes()
            results["magic number"] = head.magicNumber == 0x5F0F3CF5

            ocheck = head.checkSumAdjustment
            head.checkSumAdjustment = 0
            head_bytes2 = head.to_bytes()
            results["head checksum"] = checkSum(head_bytes2) == self.table_directory["head"].checkSum
            head.checkSumAdjustment = ocheck # reset value

            f_reconstructed = p.b[:(t:=self.table_directory["head"]).offset] + head_bytes + p.b[t.offset+t.length:]
            results["reconstruction"] = f_reconstructed == p.b

            f_for_checksum = p.b[:t.offset] + head_bytes2 + p.b[t.offset+t.length:]
            results["file checksum"] = (0xB1B0AFBA - checkSum(f_for_checksum)) == ocheck

            print(f"{'SUCCESS' if all(results.values()) else 'FAILURE'} in checking: {fontfile}", *(f"\n    {k:18s} {v}" for k,v in results.items()))

        # load tables
        self.maxp = p.parse(maxp_table, offset=self.table_directory["maxp"].offset)
        self.head = p.parse(head_table, offset=self.table_directory["head"].offset)
        self.cmap = p.parse(cmap_table, offset=self.table_directory["cmap"].offset, length=self.table_directory["cmap"].length)
        self.loca = p.parse(uint32 if self.head.indexToLocFormat else uint16, count=self.maxp.numGlyphs, offset=self.table_directory["loca"].offset)
        self.hhea = p.parse(hhea_table, offset=self.table_directory["hhea"].offset)
        self.hmtx = p.parse(hmtx_table, offset=self.table_directory["hmtx"].offset, numOfLongHorMetrics=self.hhea.numOfLongHorMetrics, numGlyphs=self.maxp.numGlyphs)
        self.post = p.parse(post_table, offset=self.table_directory["post"].offset)
        self.glyf = p.b[(t:=self.table_directory["glyf"]).offset:t.offset+t.length] # raw bytes of glyf table
        del p

        if check: return results
    
    def fupx(self, v, fontsize, dpi) -> float: return v * (fontsize * dpi) / (72 * self.head.unitsPerEM) # Font Unit to pixel conversion

    @functools.cache
    def glyph(self, unicode, fontsize, dpi):
        g = self.loadglyph(self.cmap.subtable.getGlyphIndex(unicode), fontsize, dpi)
        if g.x:
            minX, maxX = math.floor(min(g.x)), math.ceil(max(g.x))
            minY, maxY = math.floor(min(g.y)), math.ceil(max(g.y))
            size = vec2(maxX - minX, maxY - minY)
            bearing = vec2(g.leftSideBearing - (min(g.x) - minX), maxY)
        else:
            size = vec2(0, 0)
            bearing = g.leftSideBearing
        return Glyph(size, bearing, g.advanceWidth)

    def render(self, fontsize, dpi, unicode:int=None, glyphIndex=None, antialiasing=1):
        assert (unicode is not None and isinstance(unicode, int)) or (glyphIndex is not None and isinstance(glyphIndex, int))
        glyphIndex = glyphIndex if glyphIndex is not None else self.cmap.subtable.getGlyphIndex(unicode)
        return rasterize(self._canonicalize_contours(self.loadglyph(glyphIndex, fontsize, dpi)), aa=antialiasing)

    # used for compound glyphs
    @staticmethod
    def transform(v:vec2, xscale, yscale, scale01, scale10) -> vec2: return vec2(xscale * v.x + scale10 * v.y, scale01 * v.x + yscale*v.y)

    def loadglyph(self, glyphIndex, fontsize, dpi, _is_child=False) -> glyf:
        glyph = glyf(self.glyf[self.loca[glyphIndex]:self.loca[glyphIndex+1]]) if glyphIndex + 1 < len(self.loca) else glyf(self.glyf[self.loca[glyphIndex]:])
        if hasattr(glyph, "children"): # compound glyph
            for child in glyph.children: # child is glyphComponent object
                child_glyph = self.loadglyph(child.glyphIndex, fontsize, dpi, _is_child=True)
                if child.USE_MY_METRICS:
                    glyph.advanceWidth = child_glyph.advanceWidth
                    glyph.leftSideBearing = child_glyph.leftSideBearing
                xscale = yscale = 1
                scale01 = scale10 = 0
                if child.WE_HAVE_A_SCALE:
                    xscale = yscale = child.scale
                elif child.WE_HAVE_AN_X_AND_Y_SCALE:
                    xscale, yscale = child.xscale, child.yscale
                elif child.WE_HAVE_A_TWO_BY_TWO: xscale, yscale, scale01, scale10 = child.xscale, child.yscale, child.scale01, child.scale10
                pts = [self.transform(vec2(x,y), xscale, yscale, scale01, scale10) for x,y in zip(child_glyph.x, child_glyph.y)]
                child_glyph.x = [v.x for v in pts]
                child_glyph.y = [v.y for v in pts]
                if child.ARGS_ARE_XY_VALUES:
                    if child.SCALED_COMPONENT_OFFSET == child.UNSCALED_COMPONENT_OFFSET: child.UNSCALED_COMPONENT_OFFSET = True # default behaviour if flags are invalid
                    if child.UNSCALED_COMPONENT_OFFSET: offset = vec2(child.arg1, child.arg2)
                    else: offset = self.transform(vec2(child.arg1, child.arg2), xscale, yscale, scale01, scale10)
                    offset = vec2(self.fupx(child.arg1, fontsize, dpi), self.fupx(child.arg2, fontsize, dpi))
                    if child.ROUND_XY_TO_GRID: offset = vec2((offset.x + 0.5)//1, (offset.y + 0.5) // 1) # round to grid
                else: offset = vec2(glyph.x[child.arg1] - child_glyph[child.arg1], glyph.y[child.arg1] - child_glyph[child.arg2])
                # apply offset
                child_glyph.x = [x + offset.x for x in child_glyph.x]
                child_glyph.y = [y + offset.y for y in child_glyph.y]
                # recalculate endPtsContours
                if not hasattr(glyph, "endPtsContours"): setattr(glyph, "endPtsContours", child_glyph.endPtsContours)
                else: glyph.endPtsContours += [ep + glyph.endPtsContours[-1] + 1 for ep in child_glyph.endPtsContours]
                # incorporate child glyph
                if not hasattr(glyph, "x"): setattr(glyph, "x", [])
                if not hasattr(glyph, "y"): setattr(glyph, "y", [])
                if not hasattr(glyph, "flags"): setattr(glyph, "flags", [])
                glyph.x += child_glyph.x.copy()
                glyph.y += child_glyph.y.copy()
                glyph.flags += child_glyph.flags
        else: glyph.x, glyph.y = list(map(lambda v: [self.fupx(v0, fontsize, dpi) for v0 in v], [glyph.x, glyph.y])) # simple glyph: convert Funits to pixels
        if not (hasattr(glyph, "advanceWidth") and hasattr(glyph, "leftSideBearing")):
            if glyphIndex < self.hhea.numOfLongHorMetrics:
                glyph.advanceWidth = self.fupx(self.hmtx.longHorMetric[glyphIndex].advanceWidth, fontsize, dpi)
                glyph.leftSideBearing = self.fupx(self.hmtx.longHorMetric[glyphIndex].leftSideBearing, fontsize, dpi)
            else:
                glyph.advanceWidth = self.fupx(self.hmtx.longHorMetric[-1].advanceWidth, fontsize, dpi)
                glyph.leftSideBearing = self.fupx(self.hmtx.leftSideBearing[glyphIndex - self.hhea.numOfLongHorMetrics])
        
        if _is_child == False and self.head.flags & 2: glyph.x = [x - glyph.leftSideBearing for x in glyph.x] # this flag means left side bearing should be aligned with x = 0. only applies when not dealing with compound glyph components
        return glyph

    def _canonicalize_contours(self, g:glyf) -> List[List[CurvePoint]]:
        """
        Returns a 2D list of rendered pixels (Row-major).
        Uses the non-zero winding number rule: for each pixel, go right and on each intersection with any contour segment,
        determine the gradient at that point. if the gradient points up, add 1, else sub 1. If the result is zero, the point is outside, else, its inside

        because rasteriztation adds some space to the outside, it should subtract that space from bearing.x (less to move in when rendering because already moved in) and add it to bearing.y (glyph gets taller)
        """
        assert len(g.x) == len(g.y)
        if len(g.x) == 0: return []
        minX = math.floor(min(g.x))
        minY, maxY = math.floor(min(g.y)), math.ceil(max(g.y))
        g.bearing = vec2(g.leftSideBearing - (min(g.x) - minX), maxY)
        pts = [CurvePoint(x-minX, y-minY, bool(f & 0x01)) for x,y,f in zip(g.x, g.y, g.flags)] # shift into positive range
        # add oncurve points between consecutive control points
        all_contours = [] # first point will exist twice. at 0 and -1
        start = 0
        for ep in g.endPtsContours:
            new_contour = []
            contour = pts[start:ep+1]
            if contour[0].onCurve is False: # contour must start and end with onCurve for proper segment extraction in rasterization
                if contour[1].onCurve is False:
                    contour.insert(1, CurvePoint(contour[0].x+(contour[1].x-contour[0].x)/2, contour[0].y+(contour[1].y-contour[0].y)/2, True))
                contour = contour[1:] + [contour[0]] # rotate by one, so it starts with onCurve
            for p1, p2 in zip(contour, contour[1:] + [contour[0]]):
                new_contour.append(p1)
                # add another point inbetween. if two consecutive offcurve points, add an oncurve one to keep the curves quadratic. if two oncurve ones, add one so the segments are always 3 points: easy to handle.
                if p1.onCurve == p2.onCurve: new_contour.append(CurvePoint(p1.x + (p2.x-p1.x)/2, p1.y + (p2.y-p1.y) / 2, True)) 
            start = ep+1
            new_contour.append(contour[0])
            assert len(new_contour) % 2 == 1
            all_contours.append(new_contour)
        assert not any([p1.onCurve is False and p2.onCurve is False for contour in all_contours for p1,p2 in zip(contour, contour[1:] + [contour[0]])])
        return all_contours

# GENERAL CURVE RASTERIZER

@dataclass(frozen=True)
class Intersection:
    x:float
    wn:int # winding number

def rasterize(contours: List[List[CurvePoint]], aa: int) -> Tuple[List[List[int]], vec2]:
    # Uses SSAA (Super-Sampled Anti-Aliasing) if aa > 1
    assert aa >= 1
    pts = [p for c in contours for p in c]
    xs, ys = [p.x for p in pts], [p.y for p in pts]
    mx, MX, my, MY = math.floor(min(xs)), math.ceil(max(xs)), math.floor(min(ys)), math.ceil(max(ys))
    sx, sy = MX - mx, MY - my
    bmp = [[0]*sx for _ in range(sy)]

    for row in range(sy):
        all_ints = []
        for aay in range(aa):
            y = row + (aay + .5)/aa + 1e-6  # Add small bias to avoid tangency artifacts
            ints = []
            for c in contours:
                for i in range(0, len(c)-2, 2):
                    p0, p1, p2 = [vec2(pt.x, pt.y) for pt in c[i:i+3]]
                    if all(p.y < y for p in (p0,p1,p2)) or all(p.y > y for p in (p0,p1,p2)): continue
                    a, b, cc = p0.y - 2*p1.y + p2.y, 2*(p1.y-p0.y), p0.y - y
                    ts = [t for t in quadratic_equation(a,b,cc) if -1e-9 <= t <= 1+1e-9]
                    for t in ts:
                        t = max(0, min(1, t))
                        x = (1-t)**2 * p0.x + 2*(1-t)*t * p1.x + t**2 * p2.x
                        grad = vec2(2*(1-t)*(p1.x-p0.x) + 2*t*(p2.x-p1.x), 2*(1-t)*(p1.y-p0.y) + 2*t*(p2.y-p1.y))
                        if abs(grad.y) < 1e-6: wn = 0
                        else: wn = 1 if grad.y > 0 else -1
                        if not any(abs(ii.x - x) < 1e-6 and ii.wn == wn for ii in ints): ints.append(Intersection(x, wn))
            all_ints.append(ints)
            
        for col in range(sx):
            v=0
            for aax in range(aa):
                x = col + (aax+.5)/aa
                for ints in all_ints:
                    v += 1 if sum(i.wn for i in ints if i.x>x+1e-6) != 0 else 0
            bmp[row][col] = int(v/(aa**2)*255)
    return bmp

# ABSTRACT FONT INTERFACE

class Font:
    def __init__(self, path:Union[str, Path], **kwargs):
        assert (path := Path(path)).suffix == ".ttf", f"Can't load {path}. Only True Type fonts (.tff) are supported."
        self.engine = TTF(path, **kwargs)

    def glyph(self, char:str, fontsize, dpi:int) -> Glyph:
        assert len(char) == 1, f"Can't render '{char}', can only get one character at a time."
        assert fontsize > 0 and dpi > 0, f"Can't render with {fontsize=} at {dpi=}. Both values must be positive."
        return self.engine.glyph(ord(char), fontsize, dpi)

    def render(self, char:str, fontsize, dpi:int, antialiasing:int=5) -> List[List[int]]:
        assert len(char) == 1, f"Can't render '{char}', can only render one character at a time."
        assert fontsize > 0 and dpi > 0, f"Can't render with {fontsize=} at {dpi=}. Both values must be positive."
        assert antialiasing >= 1, f"Can't render with antialiasing of {antialiasing}. Must be >=1. If 1, applies no Anti-Aliasing"
        return self.engine.render(fontsize, dpi, unicode=ord(char), antialiasing=antialiasing)
    