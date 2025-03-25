from functools import partial
import math
from typing import List, Union
from enum import Enum, auto
from vec import vec2
from dtype import *
from helpers import *
from table import *

# CHECKSUM
def pad4(b:bytes) -> bytes: return b.ljust((len(b) + 3) // 4 * 4, b'\0')
def checkSum(b:bytes) -> int: return sum([int.from_bytes(b[i:i+4], "big") for i in range(0, len(b), 4) if (b:=pad4(b))]) & 0xFFFFFFFF

class RoundState(Enum):
    RTHG = auto() # round to half grid
    RTG = auto() # round to grid
    RTDG = auto() # Round to double grid
    RDTG = auto() # round down to grid
    RUTG = auto() # round up to grid
    OFF = auto() # no rounding
    SROUND = auto()
    S45ROUND = auto()

class Interpreter:
    def __init__(self, fontfile=None, test=False, debug=False):
        self.debug:bool = debug
        self.stack:bytes = None # all items on the stack are bytes, to be interpreted as each instruction wishes
        self.sp:int = None

        self.gs = { # graphics state with default values
            "auto_flip": True,
            "control_value_cut-in": 17/16, # F26Dot6
            "delta_base": 9,
            "delta_shift": 3,
            "dual_projection_vector": None,
            "freedom_vector": vec2(1, 0),
            "instruct_control": 0,
            "loop": 1,
            "minimum_distance": 1, # F26Dot6
            "projection_vector": vec2(1, 0),
            "round_state": RoundState.RTG,
            "rp0": 0,
            "rp1": 0,
            "rp2": 0,
            "scan_control": False,
            "single_width_cut_in": 0, # F26Dot6
            "single_width_value": 0, # F26Dot6
            "zp0": 1,
            "zp1": 1,
            "zp2": 1
        }
        self.scantype = None # looks like it should be in graphics state, but isn't
        self.ops = {
            0x00: partial(self.SVTCA, 0),
            0x01: partial(self.SVTCA, 1),
            0x02: partial(self.SPVTCA, 0),
            0x03: partial(self.SPVTCA, 1),
            0x04: partial(self.SFVTCA, 0),
            0x05: partial(self.SFVTCA, 1),
            0x06: partial(self.SPVTL, 0),
            0x07: partial(self.SPVTL, 1),
            0x08: partial(self.SFVTL, 0),
            0x09: partial(self.SFVTL, 1),
            0x0A: self.SPVFS,
            0x0B: self.SFVFS,
            0x0C: self.GPV,
            0x0D: self.GFV,
            0x0E: self.SFVTPV,
            0x0F: self.ISECT,
            0x10: self.SRP0,
            0x11: self.SRP1,
            0x12: self.SRP2,
            0x13: self.SZP0,
            0x14: self.SZP1,
            0x15: self.SZP2,
            0x16: self.SZPS,
            0x17: self.SLOOP,
            0x18: self.RTG,
            0x19: self.RTHG,
            0x1A: self.SMD,
            0x1B: self.ELSE,
            0x1C: self.JMPR,
            0x1D: self.SCVTCI,
            0x1E: self.SSWCI,
            0x1F: self.SSW,
            0x20: self.DUP,
            0x21: self.POP, # POP
            0x22: self.CLEAR,
            0x23: self.SWAP,
            0x24: self.DEPTH,
            0x25: self.CINDEX,
            0x26: self.MINDEX,
            0x27: self.ALIGNPTS,
            0x29: self.UTP, # not making sense yet
            0x2A: self.LOOPCALL,
            0x2B: self.CALL,
            0x2C: self.FDEF,
            0x2D: self.ENDF,
            0x2E: partial(self.MDAP, 0),
            0x2F: partial(self.MDAP, 1),
            0x30: partial(self.IUP, 0),
            0x31: partial(self.IUP, 1),
            0x32: partial(self.SHP, 0),
            0x33: partial(self.SHP, 1),
            0x34: partial(self.SHC, 0),
            0x35: partial(self.SHC, 1),
            0x36: partial(self.SHZ, 0),
            0x37: partial(self.SHZ, 1),
            0x38: self.SHPIX,
            0x39: self.IP,
            0x3A: partial(self.MSIRP, 0),
            0x3B: partial(self.MSIRP, 1),
            0x3C: self.ALIGNRP,
            0x3D: self.RTDG,
            0x3E: partial(self.MIAP, 0),
            0x3F: partial(self.MIAP, 1),
            0x40: self.NPUSHB,
            0x41: self.NPUSHW,
            0x42: self.WS,
            0x43: self.RS,
            0x44: self.WCVTP,
            0x45: self.RCVT,
            0x46: partial(self.GC, 0),
            0x47: partial(self.GC, 1),
            0x48: self.SCFS,
            0x49: partial(self.MD, 0),
            0x4A: partial(self.MD, 1),
            0x4B: self.MPPEM,
            0x4C: self.MPS,
            0x4D: self.FLIPON,
            0x4E: self.FLIPOFF,
            0x4F: self.DEBUG,
            0x50: self.LT,
            0x51: self.LTEQ,
            0x52: self.GT,
            0x53: self.GTEQ,
            0x54: self.EQ,
            0x55: self.NEQ,
            0x56: self.ODD,
            0x57: self.EVEN,
            0x58: self.IF,
            0x59: self.EIF,
            0x5A: self.AND,
            0x5B: self.OR,
            0x5C: self.NOT,
            0x5D: self.DELTAP1,
            0x5E: self.SDB,
            0x5F: self.SDS,
            0x60: self.ADD,
            0x61: self.SUB,
            0x62: self.DIV,
            0x63: self.MUL,
            0x64: self.ABS,
            0x65: self.NEG,
            0x66: self.FLOOR,
            0x67: self.CEILING,
            0x68: partial(self.ROUND, 0),
            0x69: partial(self.ROUND, 1),
            0x6A: partial(self.ROUND, 2),
            0x6B: partial(self.ROUND, 3),
            0x6C: partial(self.NROUND, 0),
            0x6D: partial(self.NROUND, 1),
            0x6E: partial(self.NROUND, 2),
            0x6F: partial(self.NROUND, 3),
            0x70: self.WCVTF,
            0x71: self.DELTAP2,
            0x72: self.DELTAP3,
            0x73: self.DELTAC1,
            0x74: self.DELTAC2,
            0x75: self.DELTAC3,
            0x76: self.SROUND,
            0x77: self.S45ROUND,
            0x78: self.JROT,
            0x79: self.JROF,
            0x7A: self.ROFF,
            0x7C: self.RUTG,
            0x7D: self.RDTG,
            0x7E: self.SANGW,
            # 0x7F: self.AA, irrelevant. not even in microsoft docs
            0x80: self.FLIPPT,
            0x81: self.FLIPRGON,
            0x82: self.FLIPRGOFF,
            0x85: self.SCANCTRL,
            0x86: partial(self.SDPVTL, 0),
            0x87: partial(self.SDPVTL, 1),
            0x88: self.GETINFO,
            0x89: self.IDEF,
            0x8A: self.ROLL,
            0x8B: self.MAX,
            0x8C: self.MIN,
            0x8D: self.SCANTYPE,
            0x8E: self.INSTCTRL,
            0xB0: partial(self.PUSHB, 0),
            0xB1: partial(self.PUSHB, 1),
            0xB2: partial(self.PUSHB, 2),
            0xB3: partial(self.PUSHB, 3),
            0xB4: partial(self.PUSHB, 4),
            0xB5: partial(self.PUSHB, 5),
            0xB6: partial(self.PUSHB, 6),
            0xB7: partial(self.PUSHB, 7),
            0xB8: partial(self.PUSHW, 0),
            0xB9: partial(self.PUSHW, 1),
            0xBA: partial(self.PUSHW, 2),
            0xBB: partial(self.PUSHW, 3),
            0xBC: partial(self.PUSHW, 4),
            0xBD: partial(self.PUSHW, 5),
            0xBE: partial(self.PUSHW, 6),
            0xBF: partial(self.PUSHW, 7),
            0xC0: partial(self.MDRP, 0),
            0xC1: partial(self.MDRP, 1),
            0xC2: partial(self.MDRP, 2),
            0xC3: partial(self.MDRP, 3),
            0xC4: partial(self.MDRP, 4),
            0xC5: partial(self.MDRP, 5),
            0xC6: partial(self.MDRP, 6),
            0xC7: partial(self.MDRP, 7),
            0xC8: partial(self.MDRP, 8),
            0xC9: partial(self.MDRP, 9),
            0xCA: partial(self.MDRP, 10),
            0xCB: partial(self.MDRP, 11),
            0xCC: partial(self.MDRP, 12),
            0xCD: partial(self.MDRP, 13),
            0xCE: partial(self.MDRP, 14),
            0xCF: partial(self.MDRP, 15),
            0xD0: partial(self.MDRP, 16),
            0xD1: partial(self.MDRP, 17),
            0xD2: partial(self.MDRP, 18),
            0xD3: partial(self.MDRP, 19),
            0xD4: partial(self.MDRP, 20),
            0xD5: partial(self.MDRP, 21),
            0xD6: partial(self.MDRP, 22),
            0xD7: partial(self.MDRP, 23),
            0xD8: partial(self.MDRP, 24),
            0xD9: partial(self.MDRP, 25),
            0xDA: partial(self.MDRP, 26),
            0xDB: partial(self.MDRP, 27),
            0xDC: partial(self.MDRP, 28),
            0xDD: partial(self.MDRP, 29),
            0xDE: partial(self.MDRP, 30),
            0xDF: partial(self.MDRP, 31),
            0xE0: partial(self.MIRP, 0),
            0xE1: partial(self.MIRP, 1),
            0xE2: partial(self.MIRP, 2),
            0xE3: partial(self.MIRP, 3),
            0xE4: partial(self.MIRP, 4),
            0xE5: partial(self.MIRP, 5),
            0xE6: partial(self.MIRP, 6),
            0xE7: partial(self.MIRP, 7),
            0xE8: partial(self.MIRP, 8),
            0xE9: partial(self.MIRP, 9),
            0xEA: partial(self.MIRP, 10),
            0xEB: partial(self.MIRP, 11),
            0xEC: partial(self.MIRP, 12),
            0xED: partial(self.MIRP, 13),
            0xEE: partial(self.MIRP, 14),
            0xEf: partial(self.MIRP, 15),
            0xF0: partial(self.MIRP, 16),
            0xF1: partial(self.MIRP, 17),
            0xF2: partial(self.MIRP, 18),
            0xF3: partial(self.MIRP, 19),
            0xF4: partial(self.MIRP, 20),
            0xF5: partial(self.MIRP, 21),
            0xF6: partial(self.MIRP, 22),
            0xF7: partial(self.MIRP, 23),
            0xF8: partial(self.MIRP, 24),
            0xF9: partial(self.MIRP, 25),
            0xFA: partial(self.MIRP, 26),
            0xFB: partial(self.MIRP, 27),
            0xFC: partial(self.MIRP, 28),
            0xFD: partial(self.MIRP, 29),
            0xFE: partial(self.MIRP, 30),
            0xFF: partial(self.MIRP, 31)
        }
        self.opcodes = {f"{v.func.__name__ if hasattr(v, 'func') else v.__name__}{v.args if hasattr(v, 'args') else ''}": k for k,v in self.ops.items()}
        self.opnames = {v: k for k,v in self.opcodes.items()}

        if fontfile: self.load(fontfile, test=test)
    
    def load(self, fontfile, fontsize=12, test=False):
        with open(fontfile, "rb") as f: 
            self.offset_subtable = offset_subtable_table(f.read(12))
            self.table_directory = {}
            self.tables = {}
            for i in range(self.offset_subtable.numTables):
                tag = f.read(4)
                entry = table_directory_entry(tag + f.read(12))
                tag = tag.decode()
                self.table_directory[tag] = entry

                pos = f.tell()
                f.seek(entry.offset)
                self.tables[tag] = f.read(entry.length)
                f.seek(pos)
            self.tables = dict(sorted(self.tables.items(), key=lambda x: self.table_directory[x[0]].offset)) # directory does not list according to order

            if test: assert all((cs:=self.checksum(f))), f"Checksum failed: {cs}"
        if self.debug: print(f"Contains tables: {list(self.tables.keys())}")

        self.context = None
        self.maxp = maxp_table(self.tables["maxp"])
        self.head = head_table(self.tables["head"])
        self.stack = [None] * self.maxp.maxStackElements
        self.sp = 0
        self.callstack = []
        self.storage = [uint32(0)] * self.maxp.maxStorage
        self.functions = [None] * self.maxp.maxFunctionDefs # contains strings like: program[o = fpgm, 1 = cvt]:instructionPointer
        self.cvt = [FWord(self.tables["cvt "][i:i+2]) for i in range(0, self.table_directory["cvt "].length, 2)] # NOTE: not a proper table object
        self.cvt = [F26Dot6(i) for i in self.cvt]
        self.twilight = [None] * self.maxp.maxTwilightPoints

        self.fontsize = fontsize
        self.dpi = 144 # TODO: get value from screen device

        self.cmap = cmap_table(self.tables["cmap"])
        p = Parser(self.tables["loca"])
        self.loca = p.parse(uint32 if self.head.indexToLocFormat else uint16, count = self.maxp.numGlyphs)
        self.g:glyf = None # stores currently loaded glyph
        self.hhea = hhea_table(self.tables["hhea"])
        self.hmtx = hmtx_table(self.tables["hmtx"], numOfLongHorMetrics=self.hhea.numOfLongHorMetrics, numGlyphs=self.maxp.numGlyphs)

        self.instruction_control = {
            "stop_grid_fit": None, # if True, no glyph programs will execute
            "default_gs": None # if True and when executing glpyh programs, use default graphics state values and ignore those set in the control value program
        }

    def checksum(self, f) -> bool:
        # check from and to bytes
        head = head_table(self.tables["head"])
        head_bytes = head.to_bytes()
        
        # check head checksum
        ocheck = head.checkSumAdjustment
        head.checkSumAdjustment = 0
        head_bytes2 = head.to_bytes()
        head.checkSumAdjustment = ocheck # reset value
        
        # reconstruct file
        f.seek(0)
        f_original = f.read()
        f_new = b''.join([self.offset_subtable.to_bytes()] + [d.to_bytes() for _, d in self.table_directory.items()] + [t + b'\x00' * ((4-len(t)%4)%4) for _,t in self.tables.items()])

        # new fle for file checksum
        newTables = self.tables.copy()
        newTables["head"] = head_bytes2
        # f_new2 = b''.join(
        #     [self.offset_subtable.to_bytes()]
        #     + [d.to_bytes() for _, d in self.table_directory.items()]
        #     + [pad4(t) for _,t in newTables.items()]
        # )
        f_new2 = b''.join([self.offset_subtable.to_bytes()] + [d.to_bytes() for _, d in self.table_directory.items()] + [t + b'\x00' * ((4-len(t)%4)%4) for _,t in newTables.items()])


        checksum_except_head = all([checkSum(v) == self.table_directory[k].checkSum for k,v in self.tables.items() if k != "head"])
        magic_numba = hex(head.magicNumber) == hex(0x5F0F3CF5)
        from_and_to_bytes = head_bytes == self.tables["head"]
        head_checksum = checkSum(head_bytes2) == self.table_directory["head"].checkSum
        reconstrucion = f_new == f_original
        file_checksum = (0xB1B0AFBA - checkSum(f_new2)) == ocheck
        # print(0xB1B0AFBA - checkSum(f_new2), ocheck)

        return (checksum_except_head, magic_numba, from_and_to_bytes, head_checksum, reconstrucion, file_checksum)

    def run(self, opcode:int):
        f = self.ops[opcode]
        if self.debug:

            fname = f"{f.func.__name__ if hasattr(f, 'func') else f.__name__}{f.args if hasattr(f, 'args') else ''}"
            if not hasattr(self, "_opset"): self._opset = set()
            self._opset.add(fname)
            self.print_debug(fname)

        f()

    def run_program(self, pgm:List[Union[bytes, int]], name:str):
        assert name in ["fpgm", "prep", "glyf"]
        self.program = pgm
        setattr(self, name, pgm)
        self.callstack = [{"context": name, "ip": 0}]
        while self.callstack[0]["ip"] < len(pgm):
            self.run(self.program[self.callstack[-1]["ip"]])
            self.callstack[-1]["ip"] += 1

    def get_program(self, context=None):
        if context == None: context = self.callstack[-1]["context"]
        match context:
            case "fpgm": return self.fpgm
            case "prep": return self.prep
            case "glyf":
                assert hasattr(self, "glyf"), "No glyph program loaded yet. use getGlyph method"
                return self.glyf

    def getGlyph(self, unicode:int):
        glyphIndex = self.cmap.subtable.getGlyphIndex(unicode)
        print(f"{glyphIndex=}")
        glyphOffset = self.loca[glyphIndex]
        glyphLength = self.loca[glyphIndex + 1] - glyphOffset
        self.g = glyf(self.tables["glyf"][glyphOffset:glyphOffset+glyphLength])
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
        
        # TODO: make points into vec2 instead of storing x and y separately
        self.g.scaled_x, self.g.scaled_y = map(lambda x: [self.FU_to_px(x0) for x0 in x], [self.g.x, self.g.y]) # convert Funits to pixels
        self.g.fit_x, self.g.fit_y = map(list.copy, (self.g.scaled_x, self.g.scaled_y))
        gs_backup = self.gs.copy()

        self.run_program(self.g.instructions, "glyf") # move stuff in self.g.fit_x and self.g.fit_y
        self.gs = gs_backup
        
        # scan-convert self.g.fit_x and self.g.fit_y to bitmap
        # return bitmap

    def FU_to_px(self, v:Union[float, int]) -> float: return v * (self.fontsize * self.dpi) / (72 * self.head.unitsPerEM)

    def orthogonal_projection(self, v1:vec2, v2:vec2) -> vec2:
        """returns vector v1 as orthogonally projected onto v2"""
        return v2 * (v1.dot(v2) / (v2.dot(v2)))
    
    def print_debug(self, op:str, v:bytes=None):
        c, ip = self.callstack[-1].values()
        depth = len(self.callstack)-1
        if op in ["PUSH", "POP"]: print(f"{'|   ' * depth}{op:4s}:[{self.sp:4}]{' 0x'+v.hex() if v != None else ''}")
        else: print(f"{'|   ' * depth}{c}:{ip} {op}{' 0x'+v.hex() if v != None else ''}")
    
    def pop(self, x=1) -> Union[int, List]:
        assert x>0 and type(x) == int
        assert self.sp != None, "No stack pointer (Interpreter.sp) set"
        ret = []
        for i in range(x):
            self.sp -= 1
            assert len(self.stack) > self.sp >= 0, f"Tried popping from outside of stack with {self.sp}"
            if self.debug: self.print_debug("POP", self.stack[self.sp])
            ret.append(self.stack[self.sp])
        return ret if len(ret) > 1 else ret[0]
         
    def push(self, *x) -> None:
        if len(x) == 1 and type(x[0]) in [list, tuple]: x = tuple(x[0])
        assert all([type(i) == bytes and len(i) == 4 for i in x]), f"Cannot only push raw bytes of length 4, got {x} instead"
        assert self.sp != None, "No stack pointer (Interpreter.sp) set"
        for i in x:
            assert 0 <= self.sp < len(self.stack), f"Tried pushing outside of stack with {self.sp}"
            if self.debug: self.print_debug("PUSH", i)
            self.stack[self.sp] = i
            self.sp += 1

    def _pushB_from_program(self, n):
        assert len(self.program) > self.callstack[-1]["ip"] + n
        for _ in range(n):
            self.callstack[-1]["ip"] += 1
            self.push(uint32.to_bytes(self.program[self.callstack[-1]["ip"]]))

    def _pushW_from_program(self, n):
        assert len(self.program) > self.callstack[-1]["ip"] + n * 2
        for _ in range(n):
            b1 = uint32(self.program[self.callstack[-1]["ip"] + 1])
            b2 = uint32(self.program[self.callstack[-1]["ip"] + 2])
            self.push(int32.to_bytes((b1 << 8) | b2))
            self.callstack[-1]["ip"] += 2

    def _pop_IS(self, n:int=1) -> int:
        """Pops and returns n bytes from instruction stream and advances the instruction pointer. Exists to avoid duplicating error checking elsewhere"""
        assert type(n) == int and n > 0, f"Invalid count {n} for popping from instruction stream"
        for _ in range(n):
            assert (self.callstack[-1]["ip"] + 1) < len(self.program), f"Tried accessing instruction outside of range of instruction stream in {self.callstack[-1]['context']} at {self.callstack[-1]['ip'] - 1}. Program already terminates at {len(self.program)}."
            self.callstack[-1]["ip"] += 1
        return uint8(self.program[self.callstack[-1]["ip"]])

    def _next_instruction(self) -> int:
        """Because some bytes in the instruction stream are not instructions but bytes to be pushed onto the stack, lookahead (as it is needed in IF ELSE and FDEF) must do some parsing lest what was meant to be a value is misinterpreted as an opcode.
        This function does this parsing. It reads the current instruction, skipps over any values associated with it and returns the next instruction while advancing the instruction pointer acoordingly."""
        # NOTE: this is reimplementing some of logic of these operations and it's stupid
        name = self.opnames[self.program[self.callstack[-1]["ip"]]]
        if name == "NPUSHB": self._pop_IS(uint8(self._pop_IS()))
        elif name == "NPUSHW": self._pop_IS(uint8(self._pop_IS()) * 2)
        elif name.startswith("PUSHB"): self._pop_IS(int(name[-3]) + 1)
        elif name.startswith("PUSHW"): self._pop_IS((int(name[-3]) + 1) * 2)
        ret = self._pop_IS()
        # print(f"_next_instruction startin from {name}, returning {ret}. {name.startswith('PUSHW')}")
        return ret

    # ----------------------------------------------------------------#
    #                                                                 #
    # INSTRUCTION IMPLEMENTATIONS                                     #
    # ----------------------------------------------------------------#
    
    def SVTCA(self, a):
        """
        Set freedom and projection Vectors To Coordinate Axis
        If a is 0, set to Y axis. If a is 1, set to X axis.
        """
        self.gs.update({"projection_vector": (v:=vec2(1, 0) if a else vec2(0, 1)), "freedom_vector": v})
    def SPVTCA(self, a):
        """
        Set Projection Vector To Coordinate Axis
        If a is 0, set to Y axis. If a is 1, set to X axis.
        """
        self.gs.update({"projection_vector": vec2(1, 0) if a else vec2(0, 1)})
    def SFVTCA(self, a):
        """
        Set Freeedom Vector To Coordinate Axis
        If a is 0, set to Y axis. If a is 1, set to X axis.
        """
        self.gs.update({"freedom_vector": vec2(1, 0) if a else vec2(0, 1)})
    def SPVTL(self, a):
        """Set Projection Vector To Line"""
        raise NotImplementedError
        #gs["projection_vector"] = vec2(-(v0:=p1-p2).y, v0.x).normalize() if a else (p1-p2).normalize() # if a rotate counter clockwise 90 deg
    def SFVTL(self, a):
        """Set Freedom Vector To Line"""
        raise NotImplementedError
        # gs["freedom_vector"] = vec2(-(v0:=p1-p2).y, v0.x).normalize() if a else (p1-p2).normalize() # if a rotate counter clockwise 90 deg
    def SPVFS(self):
        """Set Projection Vector From Stack"""
        y, x = map(lambda x: EF2Dot14(x), [self.pop(), self.pop()])
        assert math.isclose(x**2 + y**2, 1, rel_tol=1e-3)
        self.gs["projection_vector"] = vec2(x,y)
    def SFVFS(self): 
        """Set Freedom Vector From Stack"""
        y, x = map(lambda x: EF2Dot14(x), [self.pop(), self.pop()])
        assert math.isclose(x**2 + y**2, 1, rel_tol=1e-2)
        self.gs["freedom_vector"] = vec2(x,y)
    def GPV(self):
        """
        Get Projection Vector
        Interprets projection vector x and y components as EF2Dot14 and pushes first x, then y onto the stack.
        The values must be such that x**2 + y**2 (length of the vector) is 1.
        """
        x, y  = self.gs["projection_vector"].components()
        assert math.isclose(x**2 + y**2, 1), f"Vector x**2 + y**2 must equal 1, got {x=}, {y=}, {x**2 + y**2=}."
        self.push(EF2Dot14.to_bytes(x), EF2Dot14.to_bytes(y))
    def GFV(self):
        """Get Freedom Vector
        Interprets freedom vector x and y components as EF2Dot14 and pushes first x, then y onto the stack.
        The values must be such that x**2 + y**2 (length of the vector) is 1.
        """
        x, y = self.gs["freedom_vector"].components()
        assert math.isclose(x**2 + y**2, 1), f"Vector x**2 + y**2 must equal 1, got {x=}, {y=}, {x**2 + y**2=}."
        self.push(EF2Dot14.to_bytes(x), EF2Dot14.to_bytes(y))
    def SFVTPV(self):
        """Set Freedom Vector To Projection Vector"""
        self.gs["freedom_vector"] = self.gs["projection_vector"]
    def ISECT(self): raise NotImplementedError
    def SRP0(self):
        """
        Set Reference Point 0
        Pops uint32 from stack, sets reference point 0 in graphics state to that number.
        """
        self.gs["rp0"] = uint32(self.pop())
    def SRP1(self):
        """
        Set Reference Point 1
        Pops uint32 from stack, sets reference point 1 in graphics state to that number.
        """
        self.gs["rp1"] = uint32(self.pop())
    def SRP2(self):
        """
        Set Reference Point 2
        Pops uint32 from stack, sets reference point 2 in graphics state to that number.
        """
        self.gs["rp2"] = uint32(self.pop())
    def SZP0(self):
        """
        Set Zone Pointer 0
        Pops uint32 from stack, sets zone pointer 0 in graphics state to that number. Number must be 0 or 1.
        """
        assert (a:=uint32(self.pop())) in [0,1]
        self.gs["zp0"] = a
    def SZP1(self): 
        """
        Set Zone Pointer 1
        Pops uint32 from stack, sets zone pointer 1 in graphics state to that number. Number must be 0 or 1.
        """
        assert (a:=uint32(self.pop())) in [0,1]
        self.gs["zp1"] = a
    def SZP2(self):
        """
        Set Zone Pointer 2
        Pops uint32 from stack, sets zone pointer 2 in graphics state to that number. Number must be 0 or 1.
        """
        assert (a:=uint32(self.pop())) in [0,1]
        self.gs["zp2"] = a
    def SZPS(self):
        """
        Set Zone Pointers
        Pops uint32 from stack, sets all zone pointers in graphcis state to that number. Number must be 0 or 1.
        """
        assert (a:=uint32(self.pop())) in [0,1]
        self.gs["zp0"] = self.gs["zp1"] = self.gs["zp2"] = a
    def SLOOP(self):
        """
        Set LOOP variable
        Pops integer from stack, sets loop in graphics state to that number.
        """
        assert(a:=int32(self.pop())) > 0
        self.gs["loop"] = a
    def RTG(self):
        """
        Round To Grid
        Sets the round_state in graphics state to 1. Distances are rounded to the closest grid line.
        """
        self.gs["round_state"] = RoundState.RTG
    def RTHG(self):
        """
        Round To Grid
        Sets the round_state in graphics state to 0. Distances are rounded to the closest half grid line.
        """
        self.gs["round_state"] = RoundState.RTHG
    def SMD(self):
        """
        Set Minimum Distance
        Pops F26Dot6 from the stack, sets the minimum_distance in graphics state to that value. The distance is assumed to be expressed in sixty-fourths of a pixel. 
        """
        self.gs["minimum_distance"] = F26Dot6(self.pop())
    def ELSE(self): # skip to EIF
        """
        Marks the start of the sequence of instructions that are to be executed if an IF instruction encounters a FALSE value on the stack. This sequence of instructions is terminated with an EIF instruction. ELSE is only executed directly if the IF test was TRUE and the ELSE section should be skipped. If IF test was FALSE, then this instruction is skipped.
        """
        lvl = 0
        while (name:=self.opnames[self._next_instruction()]) != "EIF" or lvl != 0:
            if name == "IF": lvl += 1
            elif name == "EIF": lvl -= 1
        self.run(self.opcodes["EIF"]) # for debugging purposes only. TODO: remove shit like this
    def JMPR(self):
        """
        Jump relative
        Pops an integer offset from the stack. The signed offset is added to the instruction pointer value and execution resumes from that pointer.
        """
        self.sp+=int32(self.pop()) - 1 # after this instruction ip will increment by 1 automatically
    def SCVTCI(self):
        """
        Set Control Value Table Cut-in
        Pops an F26Dot6 from the stack, sets control value cut-in in the graphics state to that value.
        """
        self.gs["control_value_cut-in"] = F26Dot6(self.pop())
    def SSWCI(self):
        """
        Set Single Width Cut-in
        Pops an F26Dot6 from the stack, sets single width cut-in in the graphics state to that value.
        """
        self.gs["single_width_cut_in"] = self.pop()
    def SSW(self):
        """
        Set Single-width
        Pops an F26Dot6 from the stack, which is assumed to be in FUnits(, converts it to pixels on the current scale?) and sets single_width_value in the graphics state to that new value
        """
        self.gs["single_width_value"] = F26Dot6(self.pop())
    def DUP(self):
        """Duplicate top stack element"""
        self.push((a:=self.pop()), a)
    def POP(self):
        """Pop top stack element"""
        self.pop()
    def CLEAR(self):
        """CLEAR the entire stack"""
        self.stack, self.sp = [None] * self.maxp.maxStackElements, 0
    def SWAP(self):
        """SWAP the top two elements on the stack"""
        a1, a2 = self.pop(), self.pop()
        self.push(a1); self.push(a2)
    def DEPTH(self):
        """Push Depth of the stack"""
        self.push(uint32.to_bytes(self.sp))
    def CINDEX(self):
        """
        Copy the INDEXed element to the top of the stack
        Pop a uint32 index from the stack, copy the nth element counting from the top of the stack and push it on the top.
        """
        a = uint32(self.pop())
        i = self.sp - a
        assert self.sp > i >= 0, f"invalid index: {i}"
        self.push(self.stack[i])
    def MINDEX(self):
        """
        Move the INDEXed element to the top of the stack
        Same as CINDEX except the value is not copied up, but moved up instead.
        """
        a = uint32(self.pop())
        i = self.sp - a
        assert self.sp > i >= 0, f"invalid index: {i}"
        v = self.stack[i]
        for j in range(i, self.sp-1): self.stack[j] = self.stack[j+1]
        self.pop() # everything is shifted down by 1, so stack depth is reduced
        self.push(v)
    def ALIGNPTS(self): raise NotImplementedError # TODO: access glyph points
    def UTP(self): raise NotImplementedError
    def LOOPCALL(self):
        """
        Pop uint32 (i) from the stack, which is the number of a function. i must not exceed the maximum number of functions as defined in maxp table. Pop another uint32, which is the number of times the function i should be called.
        """
        f, count = uint32(self.pop()), uint32(self.pop())
        assert count >= 0
        for i in range(count):
            self.push(uint32.to_bytes(f)) # produces a push line in debug mode that looks like it belongs to the pervious op
            self.run(self.opcodes["CALL"])
    def CALL(self):
        """
        Pop uint32 (i) from the stack, which is the id of a function. i must not exceed the maximum number of functions as defined in maxp table. Call that function.
        """
        f = uint32(self.pop())
        assert self.maxp.maxFunctionDefs > f >= 0
        self.callstack.append(self.functions[f].copy())
        self.program = self.get_program()
        # print(self.callstack)
        while self.program[self.callstack[-1]["ip"]] != self.opcodes["ENDF"]:
            self.run(self.program[self.callstack[-1]["ip"]])
            self.callstack[-1]["ip"] += 1
        # print("ended function")
        self.run(self.opcodes["ENDF"])
    def FDEF(self):
        """
        Function DEFinition
        Marks the start of a function definition. Pops a uint32 that becomes the id of the function to be defined. This id must not exceed the maximum number of functions as defined in maxp table.
        Functions can only be defined in the fpgm or prep program, not in the glyph program. Function definitions may not be nested.
        """
        assert self.callstack[-1]["context"] in ["fpgm", "prep"], "Disallowed: function definition inside glyph program"
        f = uint32(self.pop())
        assert self.maxp.maxFunctionDefs > f >= 0
        op = self._next_instruction()
        self.functions[f] = self.callstack[-1].copy() # after advancing instruction pointer so that calling the function will not cause another function definition
        while (name:=self.opnames[op]) != "ENDF":
            assert name != "FDEF", "Error: found nested function, which are not supported"
            op = self._next_instruction()
    def ENDF(self):
        """END Function definition"""
        self.callstack.pop()
        self.program = self.get_program()
    def MDAP(self, a): raise NotImplementedError
    def IUP(self, a): raise NotImplementedError
    def SHP(self, a): raise NotImplementedError
    def SHC(self, a): raise NotImplementedError
    def SHZ(self, a): raise NotImplementedError
    def SHPIX(self): raise NotImplementedError
    def IP(self): raise NotImplementedError
    def MSIRP(self, a): raise NotImplementedError
    def ALIGNRP(self): raise NotImplementedError
    def RTDG(self):
        """
        Round to Double Grid
        Sets the round_state variable in the graphics state to 2. Distances are rounded to the closest half or integer pixel.
        """
        self.gs["round_state"] = RoundState.RTDG
    def MIAP(self, a): raise NotImplementedError
    def NPUSHB(self):
        """
        PUSH N Bytes
        Take 1 unsigned byte n from the instruction stream. Proceed to take n bytes from the instruction stream one after the other, pad them to uint32 and push onto the stack.
        """
        self.callstack[-1]["ip"] += 1
        n = uint8(self.program[self.callstack[-1]["ip"]])
        if self.debug: self.print_debug("POP IS", uint8.to_bytes(n))
        assert len(self.program) > self.callstack[-1]["ip"] + n
        self._pushB_from_program(n)
    def NPUSHW(self): 
        """
        PUSH N Words
        Take 1 unsigned byte n from the instruction stream. Proceed to take n words from the instruction stream one after the other, sign extend them to int32 and push onto the stack. The high byte of each word appears first.
        """
        self.callstack[-1]["ip"] += 1
        n = uint8(self.program[self.callstack[-1]["ip"]])
        if self.debug: self.print_debug("POP IS", uint8.to_bytes(n))
        assert len(self.program) > self.callstack[-1]["ip"] + n * 2
        self._pushW_from_program(n)
    def WS(self):
        """Write Store.
        Pop uint32 (value) from the stack. Pop uint32 (storage area location). It must not exceed the maxStorage as defined in the maxp table. Set the storage at that location to the previously popped value.
        Always remember that the Apple True Type specification sucks for claiming that index should be popped before stack instead of the other way around!
        """
        v = uint32(self.pop())
        assert 0 <= (i:=uint32(self.pop())) < self.maxp.maxStorage
        self.storage[i] = v
    def RS(self):
        """
        Read Store
        Pop uint32 (storage area location). It must not exceed the maxStorage as defined in the maxp table. Take the value at that storage location and push it onto the stack.
        """
        assert 0 <= (i:=uint32(self.pop())) < self.maxp.maxStorage
        self.push(uint32.to_bytes(self.storage[i]))
    def WCVTP(self):
        """
        Write Control Value Table in Pixel units
        Pop an F26Dot6 (value) and a uint32 (index) from the stack. Index must not exceed the maximum number of control value table elements.
        It is assumed that the value is in pixel units, not in FUnits. The value is put into the cvt table without modifying it.
        """
        v, i = F26Dot6(self.pop()), uint32(self.pop())
        assert 0 <= i < len(self.cvt)
        self.cvt[i] = v
    def RCVT(self):
        """
        Read Control Value Table entry
        Pop a uint32 (index) from the stack. Index must not exceed the maximum number of control value table elements. Retrieve the value at this index from the control value table and push the value onto the stack.
        """
        assert 0 <= (i:=uint32(self.pop())) < len(self.cvt)
        self.push(F26Dot6.to_bytes(self.cvt[i]))
    def GC(self, a): raise NotImplementedError
    #     """Get Coordinate projected onto the projection vector"""
    #     pi1 = uint32(self.pop())
    #     assert self.gs["zp2"] == 1 and a == 1, "Not implemented"
    #     p1 = self.g.get_point(pi1)
    #     v = self.orthogonal_projection(vec2(p1.x, p1.y), self.gs["projection_vector"]).mag()
    #     self.push(F26Dot6.to_bytes(v))
    def SCFS(self): raise NotImplementedError
    def MD(self, a):
        """
        Measure distance
        Pop two uint32 (point indices p1 and p2) from the stack. p1 uses zp1, p2 uses zp0. Project the points onto the projection vector. Then measure distance from point 1 to point 2 (point 2 - point 1). The distance is measured in pixels and pushed onto the stack as a F26Dot6.
        If a is 0, the distance is measured in the grid fitted outline, if a is 1 its measured in the original outline.
        """
        pi1, pi2 = uint32(self.pop()), uint32(self.pop())
        assert hasattr(self, "g"), "No glyph loaded"
        p1 = self.g.get_point(pi1) if self.gs["zp1"] == 1 else self.twilight[pi1]
        p2 = self.g.get_point(pi2) if self.gs["zp0"] == 1 else self.twilight[pi2]
        if a == 0: raise NotImplementedError
        elif a == 1:
            p1p = self.orthogonal_projection(p1, self.gs["projection_vector"])
            p2p = self.orthogonal_projection(p2, self.gs["projection_vector"])
            self.push(F26Dot6.to_bytes(self.FU_to_px((p2p - p1p).mag())))
        # project onto projection vector.
    def MPPEM(self):
        """Measure Pixels Per EM"""
        # ignoring direction of projection vector, so scaling is linear in all directions.
        self.push(uint32.to_bytes(int((self.fontsize * self.dpi) / 72)))
    def MPS(self): raise NotImplementedError
    def FLIPON(self): self.gs["auto_flip"] = True
    def FLIPOFF(self): self.gs["auto_flip"] = False
    def DEBUG(self): raise NotImplementedError
    def LT(self):
        """Less than"""
        self.push(uint32.to_bytes(int(uint32(self.pop()) > uint32(self.pop()))))
    def LTEQ(self):
        """Less than or equal"""
        self.push(uint32.to_bytes(int(uint32(self.pop()) >= uint32(self.pop()))))
    def GT(self):
        """Greater than"""
        self.push(uint32.to_bytes(int(uint32(self.pop()) < uint32(self.pop()))))
    def GTEQ(self):
        """Greater than or equal"""
        self.push(uint32.to_bytes(int(uint32(self.pop()) <= uint32(self.pop()))))
    def EQ(self):
        """Equal"""
        self.push(uint32.to_bytes(int(uint32(self.pop()) == uint32(self.pop()))))
    def NEQ(self):
        """Not equal"""
        self.push(uint32.to_bytes(int(uint32(self.pop()) != uint32(self.pop()))))
    def ODD(self): raise NotImplementedError
    def EVEN(self): raise NotImplementedError
    def IF(self):
        """
        IF test
        Pops an integer, e, from the stack. If e is zero (FALSE), the instruction pointer is moved to the associated ELSE or EIF[] instruction in the instruction stream. If e is nonzero (TRUE), the next instruction in the instruction stream is executed. Execution continues until the associated ELSE[] instruction is encountered or the associated EIF[] instruction ends the IF[] statement. If an associated ELSE[] statement is found before the EIF[], the instruction pointer is moved to the EIF[] statement.
        """
        if not int32(self.pop()):
            lvl = 0
            op = self._next_instruction()
            while (name:=self.opnames[op]) not in ["ELSE", "EIF"] or lvl != 0:
                depth = 0
                if name == "FDEF": depth += 1
                if name == "ENDF":
                    assert depth > 0, "Error: ended function definition before if statement was finished."
                    depth -= 1
                if name == "IF": lvl += 1
                elif name == "EIF": lvl -= 1
                op = self._next_instruction()
            if name == "ELSE": # lookahead without moving instruction pointer to verify that there is an end to the IF statement
            # NOTE: deeply nested if statement make this run over the same shit as many times as its deep.
                original_index = self.callstack[-1]["ip"] # to reset ip later
                lvl = depth = 0
                while (name:=self.opnames[self._next_instruction()]) != "EIF" or lvl !=0 :
                    assert not (lvl == 0 and name == "ELSE"), f"Error: found ELSE in {self.callstack[-1]['context']} at {self.callstack[-1]['ip']} following another ELSE at {original_index}"
                    if name == "FDEF": depth += 1
                    elif name == "ENDF":
                        assert depth > 0, "Error: ended function definition before if statement was finished."
                        depth -= 1
                    elif name == "IF": lvl += 1
                    elif name == "EIF": lvl -= 1
                self.callstack[-1]["ip"] = original_index

    def EIF(self):
        """
        End IF
        marks the end of an IF instruction
        """
    def AND(self):
        """Logical AND"""
        v1, v2 = uint32(self.pop()), uint32(self.pop())
        self.push(uint32.to_bytes(int(bool(v1) and bool(v2))))
    def OR(self):
        """Logical OR"""
        v1, v2 = uint32(self.pop()), uint32(self.pop())
        self.push(uint32.to_bytes(int(bool(v1) or bool(v2))))
    def NOT(self):
        """Logical NOT"""
        v = uint32(self.pop())
        self.push(uint32.to_bytes(int(not bool(v))))
    def DELTAP1(self): raise NotImplementedError
    def SDB(self):
        """Set Delta_Base in the graphics state"""
        self.gs["delta_base"] = uint32(self.pop())
    def SDS(self):
        """Set Delta_Shift in the graphics state"""
        self.gs["delta_shift"] = uint32(self.pop())
    def ADD(self):self.push(F26Dot6.to_bytes(F26Dot6(self.pop()) + F26Dot6(self.pop())))
    def SUB(self): self.push(F26Dot6.to_bytes(-F26Dot6(self.pop()) + F26Dot6(self.pop())))
    def DIV(self):
        a, b = F26Dot6(self.pop()), F26Dot6(self.pop())
        assert a != 0, f"Error: division by zero in {self.callstack[-1]['context']} instruction {self.callstack[-1]['ip']}"
        self.push(F26Dot6.to_bytes(b / a))
    def MUL(self):
        a, b = F26Dot6(self.pop()), F26Dot6(self.pop())
        self.push(F26Dot6.to_bytes(a * b))
    def ABS(self):
        """Absoluet value"""
        v = F26Dot6(self.pop())
        self.push(F26Dot6.to_bytes(v if v >= 0 else -v))
    def NEG(self):
        """Negate"""
        self.push(F26Dot6.to_bytes(-F26Dot6(self.pop())))
    def FLOOR(self): self.push(F26Dot6.to_bytes(math.floor(F26Dot6(self.pop()))))
    def CEILING(self): self.push(F26Dot6.to_bytes(math.ceil(F26Dot6(self.pop()))))
    def ROUND(self, a): raise NotImplementedError
    def NROUND(self, a): raise NotImplementedError
    def WCVTF(self):
        """Write Control Value Table in Funits"""
        v = uint32(self.pop()) # FUnits
        self.cvt[uint32(self.pop())] = F26Dot6(self.FU_to_px(v)) # pixels
    def DELTAP2(self): raise NotImplementedError
    def DELTAP3(self): raise NotImplementedError
    def DELTAC1(self): raise NotImplementedError
    def DELTAC2(self): raise NotImplementedError
    def DELTAC3(self): raise NotImplementedError
    def SROUND(self): raise NotImplementedError
    def S45ROUND(self): raise NotImplementedError
    def JROT(self): raise NotImplementedError
    def JROF(self): raise NotImplementedError
    def ROFF(self): self.gs["round_state"] = RoundState.OFF
    def RUTG(self): self.gs["round_state"] = RoundState.RUTG
    def RDTG(self): self.gs["round_state"] = RoundState.RDTG
    def SANGW(self): raise NotImplementedError
    def AA(self): raise NotImplementedError
    def FLIPPT(self): raise NotImplementedError
    def FLIPRGON(self): raise NotImplementedError
    def FLIPRGOFF(self): raise NotImplementedError
    def SCANCTRL(self): # probably just set the value of scan_control and interpret the bits when starting scan converter
        """Scan conversion control"""
        self.pop()
        self.gs["scan_control"] = True
        # self.gs["scan_control"] = Euint16(self.pop())
    def SDPVTL(self, a): raise NotImplementedError
    def GETINFO(self):
        i = uint32(self.pop())
        result = 0
        if i & 0x00000001: result |= 0x00000002 # scaler version. I'm pretending to be Macintosh System 7
        if i & 0x00000002: result |= 0x00000000 # glyph rotation
        if i & 0x00000004: result |= 0x00000000 # glyph stretched
        if i & 0x00000008: result |= 0x00000000 # font variations
        if i & 0x00000010: result |= 0x00000000 # Vertical Phantom Points
        if i & 0x00000020: result |= 0x00000000 # Windows Font Smoothing Grayscale
        if i & 0x00000040: result |= 0x00000000 # ClearType enabled
        if i & 0x00000080: result |= 0x00000000 # ClearType compatible widths enabled
        if i & 0x00000100: result |= 0x00000000 # ClearType Horizontal LCD stripe orientation
        if i & 0x00000200: result |= 0x00000000 # ClearType BGR LCD stripe order
        if i & 0x00000800: result |= 0x00000000 # ClearType sub-pixel poisitoned text enabled
        if i & 0x00000400: result |= 0x00000000 # ClearType symmetric rendering enabled
        if i & 0x00001000: result |= 0x00000000 # ClearType Gray rendering enabled
        self.push(uint32.to_bytes(result))
    def IDEF(self): raise NotImplementedError
    def ROLL(self):
        self.push(uint32.to_bytes(3))
        self.MINDEX()
    def MAX(self): raise NotImplementedError #self.push(max(self.pop(), self.pop()))
    def MIN(self): raise NotImplementedError # self.push(min(self.pop(), self.pop()))
    def SCANTYPE(self): self.scantype = Euint16(self.pop())
    def INSTCTRL(self):
        """
        Instruction execution control

        """
        selector = int32(self.pop())
        assert self.callstack[-1]["context"] == "prep" or selector == 3
        v = Euint16(self.pop())
        assert selector in [1,2,3]
        if selector == 1:
            assert v in [0,1]
            self.instruction_control["stop_grid_fit"] = bool(v)
        elif selector == 2:
            assert v in [0,2]
            self.instruction_control["default_gs"] = bool(v)
        elif selector == 3: pass # ClearType related, idk
    def PUSHB(self, a): self._pushB_from_program(a + 1)
    def PUSHW(self, a): self._pushW_from_program(a + 1)
    def MDRP(self, a): raise NotImplementedError
    def MIRP(self, a): raise NotImplementedError