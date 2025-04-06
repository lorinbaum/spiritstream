from typing import List, Union, Dict
from vec import vec2
from dtype import *
from table import *
import op as ops

# Architecture notes:
# - have a ttf file object that manages table data access efficiently

# CHECKSUM
def pad4(b:bytes) -> bytes: return b.ljust((len(b) + 3) // 4 * 4, b'\0')
def checkSum(b:bytes) -> int: return sum([int.from_bytes(b[i:i+4], "big") for i in range(0, len(b), 4) if (b:=pad4(b))]) & 0xFFFFFFFF

class Interpreter:
    def __init__(self, fontfile=None, debug=False):
        self.debug:bool = debug
        if fontfile: self.load(fontfile)
    
    def load(self, fontfile):
        # load tables
        with open(fontfile, "rb") as f: p = Parser(f.read())
        self.offset_subtable = p.parse(offset_subtable_table)
        self.table_directory = {uint32.to_bytes(entry.tag).decode():entry for entry in p.parse(table_directory_entry, count=self.offset_subtable.numTables)}
        self.maxp = p.parse(maxp_table, offset=self.table_directory["maxp"].offset)
        self.head = p.parse(head_table, offset=self.table_directory["head"].offset)
        self.original_cvt = p.parse(FWord, count=self.table_directory["cvt "].length, offset=self.table_directory["cvt "].offset)
        self.cmap = p.parse(cmap_table, offset=self.table_directory["cmap"].offset, length=self.table_directory["cmap"].length)
        self.loca = p.parse(uint32 if self.head.indexToLocFormat else uint16, count=self.maxp.numGlyphs, offset=self.table_directory["loca"].offset)
        self.hhea = p.parse(hhea_table, offset=self.table_directory["hhea"].offset)
        self.hmtx = p.parse(hmtx_table, offset=self.table_directory["hmtx"].offset, numOfLongHorMetrics=self.hhea.numOfLongHorMetrics, numGlyphs=self.maxp.numGlyphs)
        self.glyf = p.b[(t:=self.table_directory["glyf"]).offset:t.offset+t.length] # raw bytes of glyf table
        self.fpgm = p.b[(t:=self.table_directory["fpgm"]).offset:t.offset+t.length]
        self.prep = p.b[(t:=self.table_directory["prep"]).offset:t.offset+t.length]
        if self.debug:
            print(f"Contains tables: {list(self.table_directory.keys())}")
            assert all((cs:=self.checksum(p.b))), f"Checksum failed: {cs}"
        del p

        # set up state
        self.gs = { # graphics state with default values
            "auto_flip": True,
            "control_value_cut_in": 17/16, # F26Dot6
            "delta_base": 9,
            "delta_shift": 3,
            "dual_projection_vector": None,
            "freedom_vector": vec2(1, 0),
            "instruct_control": 0,
            "loop": 1,
            "minimum_distance": 1, # F26Dot6
            "projection_vector": vec2(1, 0),
            "round_state": {"period": 1, "phase": 0, "threshold": 0.5},
            "rp0": 0, "rp1": 0, "rp2": 0,
            "scan_control": False,
            "single_width_cut_in": 0, # F26Dot6
            "single_width_value": 0, # F26Dot6
            "zp0": 1, "zp1": 1, "zp2": 1
        }
        self.scantype = None # feels like it should be in graphics state, but isn't
        self.stack = [None] * self.maxp.maxStackElements
        self.sp = 0
        self.callstack = []
        self.storage = [0] * self.maxp.maxStorage
        self.functions:List[Dict["program":List[int], int]] = [None] * self.maxp.maxFunctionDefs
        # self.cvt = self.original_cvt.copy()
        self.g:glyf = None # stores currently loaded glyph

        self.fontsiz = self.dpi = None # TODO: get value from screen device

        self.instruction_control = {
            "stop_grid_fit": None, # if True, no glyph programs will execute
            "default_gs": None # if True and when executing glpyh programs, use default graphics state values and ignore those set in the control value program
        }
        self.run(self.fpgm)

    def checksum(self, f_original:bytes) -> bool:
        head_bytes = f_original[(t:=self.table_directory["head"]).offset:t.offset+t.length]
        head = head_table(head_bytes)
        from_and_to_bytes = head_bytes == head.to_bytes()
        magic_number = head.magicNumber == 0x5F0F3CF5

        ocheck = head.checkSumAdjustment
        head.checkSumAdjustment = 0
        head_bytes2 = head.to_bytes()
        head_checksum = checkSum(head_bytes2) == self.table_directory["head"].checkSum
        head.checkSumAdjustment = ocheck # reset value

        f_reconstructed = f_original[:(t:=self.table_directory["head"]).offset] + head_bytes + f_original[t.offset+t.length:]
        reconstrucion = f_reconstructed == f_original

        f_for_checksum = f_original[:t.offset] + head_bytes2 + f_original[t.offset+t.length:]
        file_checksum = (0xB1B0AFBA - checkSum(f_for_checksum)) == ocheck

        return (from_and_to_bytes, magic_number, head_checksum, reconstrucion, file_checksum)
    
    def run(self, pgm:Union[int, str, bytes, List[Union[int, str]]]):
        """Takes a program as a list of opcodes or opnames or a single stream of bytes or a single opcode or opname. Runs that program. If the end of the program is reached, it automatically ends without raising an error."""
        if isinstance(pgm, (int, str)): pgm = [pgm]
        assert all([isinstance(op, (int, str)) for op in pgm])
        pgm = b''.join([uint8.to_bytes(ops.name_code[op] if isinstance(op, str) else op) for op in pgm]) # programs are always bytestreams for consistency
        self.callstack.append({"program": pgm, "ip": -1})
        if len(self.callstack) == 1: # maintain the loop only for the first call instead of recursively entering loops
            while len(self.callstack) > 0:
                (c:=self.callstack[-1])["ip"] += 1
                if c["ip"] < len(c["program"]):
                    op = ops.code_op[el:=uint8(c["program"][c["ip"]])]
                    if self.debug:
                        if not hasattr(self, "_opset"): self._opset = set()
                        self._opset.add(name:=ops.code_name[el])
                        self.print_debug(name)
                    try: op(self)
                    except Exception:
                        self.callstack = [] # For tests. If they do assertRaises and proceed to run other programs, the callstack should be reset
                        raise
                else: self.callstack.pop()
    
    def get_point(self, point_index:int, zone_pointer:str, outline:str="original"):
        assert zone_pointer in ["zp0", "zp1", "zp2"]
        zp = self.gs[zone_pointer]
        if zp == 0: return self.twilight[point_index]
        else: return self.g.get_point(point_index, outline)

    def getGlyph(self, unicode:int, fontsize=None, dpi=None):
        assert fontsize != None or self.fontsize != None, "Specify fontsize"
        assert dpi != None or self.dpi != None, "Specify dpi"
        if dpi == None: dpi = self.dpi
        if fontsize == None: fontsize = self.fontsize
        if (not hasattr(self, "fontsize") or fontsize != self.fontsize) or (not hasattr(self, "dpi") or dpi != self.dpi):
            self.fontsize, self.dpi = fontsize, dpi
            self.twilight = [vec2(0,0)] * self.maxp.maxTwilightPoints
            # reset cvt table
            self.cvt = self.original_cvt.copy()
            self.run(self.prep)
            # scale cvt
            self.cvt = [F26Dot6(ops.FU_to_px(self, v)) for v in self.cvt]

        glyphIndex = self.cmap.subtable.getGlyphIndex(unicode)
        glyphOffset = self.loca[glyphIndex]
        glyphLength = self.loca[glyphIndex + 1] - glyphOffset
        self.g = glyf(self.glyf[glyphOffset:glyphOffset+glyphLength])
        # create phantom points (origin and advance points)
        if self.head.flags & 1: baseline = 0 # y = 0
        else: raise NotImplementedError
        if glyphIndex < self.hhea.numOfLongHorMetrics:
            advanceWidth = self.hmtx.longHorMetric[glyphIndex].advanceWidth
            leftSideBearing = self.hmtx.longHorMetric[glyphIndex].leftSideBearing
        else:
            advanceWidth = self.hmtx.longHorMetric[-1].advanceWidth
            leftSideBearing = self.hmtx.leftSideBearing[glyphIndex - self.hhea.numOfLongHorMetrics]
        if self.head.flags & 2: assert self.g.xMin == leftSideBearing
        self.g.x += [leftSideBearing, leftSideBearing + advanceWidth]
        self.g.y += [baseline, baseline]
        
        self.g.scaled_x, self.g.scaled_y = list(map(lambda v: [ops.FU_to_px(self, v0) for v0 in v], [self.g.x, self.g.y])) # convert Funits to pixels
        self.g.fitted_x, self.g.fitted_y = self.g.scaled_x.copy(), self.g.scaled_y.copy()
        self.g.touched = [False] * len(self.g.x) # touched / untouched matters in some instructions (IUP)

        gs_backup = self.gs.copy()
        self.run(self.g.instructions) # move stuff in self.g.fit_x and self.g.fit_y
        self.gs = gs_backup
        
        # scan-convert self.g.fit_x and self.g.fit_y to bitmap
        # return bitmap
    
    def print_debug(self, op:str, v:bytes=None):
        _, ip = self.callstack[-1].values()
        depth = len(self.callstack)-1
        if op in ["PUSH", "POP"]: print(f"{'|   ' * depth}{op:4s}:[{self.sp:4}]{' 0x'+v.hex() if v != None else ''}")
        else: print(f"{'|   ' * depth}{ip} {op}{' 0x'+v.hex() if v != None else ''}")