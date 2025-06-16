from spiritstream.dtype import *
from typing import Union
from spiritstream.vec import vec2
import math

class Parser:
    def __init__(self, buffer:bytes):
        self.b = buffer
        self.p = 0 # pointer
    def parse(self, dt:DType, count:int=1, offset:int=None, **kwargs) -> int: # dtype incorrect, should be Union[...] and include table
        assert count >= 0, f"Got invalid count {count}"
        if count == 0: return []
        ret = []
        if offset: self.p = offset
        for i in range(count):
            if hasattr(dt, "size"): # dtypes have size, tables don't
                assert len(self.b) >= self.p + dt.size
                ret.append(dt(self.b[self.p : self.p+dt.size]))
                self.p += dt.size
            else: ret.append(dt(self, **kwargs))# self.p will still increment because all tables are made from dtypes and they have sizes. kwargs are needed for hmtx table
        return ret if len(ret) > 1 else ret[0]

class Serializer:
    """Given a buffer of bytes, it's method serialize can be called with a dtype and a value and it will serialize it into the buffer. It exists as a wrapper around the buffer to automatically increment the pointer"""
    def __init__(self, size:int):
        self.size = size
        self.b = b''
    def serialize(self, dt:DType, v:Union[int, float]): # works only with dtypes, not tables
        self.b += dt.to_bytes(v)
        assert len(self.b) <= self.size, f"Error: Exceeded expected buffer size. {self.size=}"
    
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
                        if unicode <= self.endCode[i]:
                            if self.startCode[i] <= unicode:
                                glyphIndex = int((i + self.idRangeOffset[i] / 2 + (unicode - self.startCode[i])) - len(self.idRangeOffset))
                                # v = self.getGlyphIndex[glyphIndex]
                                v = self.glyphIdArray[glyphIndex]
                                if v != 0: return (self.idDelta[i] + v) % 65536
                                elif self.idRangeOffset[i] == 0: return (self.idDelta[i] + unicode) & 65536
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
    class glyphPoint(vec2):
        def __init__(self, x:int, y:int, onCurve:bool):
            self.onCurve = onCurve
            self.x, self.y = x, y
        def __repr__(self): return f"glyphPoint({self.x=}, {self.y=}, {self.onCurve=})"
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
    
    def get_point(self, i:int) -> glyphPoint:
        assert len(self.x) > i >= 0, f"Error: invalid index {i=}"
        return self.glyphPoint(self.x[i], self.y[i], bool(self.flags[i] & 0x01))

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