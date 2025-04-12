import unittest, math
from ttf import *
from random import random, randint
from typing import List, Union
from table import glyf
from op import push, pop, orthogonal_projection, FU_to_px, name_code

class maxp:
    maxFunctionDefs = 2
    maxStorage = 2
    maxStackElements = 4

class head:
    unitsPerEM = 2048

class glyf(glyf):
    def __init__(self): # override inherited init to create class instance without bytse to interpret
        self.x = [0, 1, 5, 0, 1, 1.2]
        self.y = [0, 2,-10, 3, 100, 0.0012]
        self.flags = [0, 1, 1, 0, 0, -5]

twilight_points = [vec2(0,0), vec2(0,1), vec2(1,0), vec2(1,1), vec2(-1,0), vec2(0,-1), vec2(-1,-1), vec2(12.123, -1239), vec2(-0, 129312)]

def uint32_stack(v:List[Union[int, float]]): return [uint32.to_bytes(i) if i != None else i for i in v]
def int32_stack(v:List[Union[int, float]]): return [int32.to_bytes(i) if i != None else i for i in v]
# TODO: looking up opcodes / opnames should not require an instance of Interpreter
# def to_program(I:Interpreter, i:List[Union[str, int]]):
#     return b''.join([uint8.to_bytes(I.opcodes[name] if type(name) == str else uint8.to_bytes(name) if name >= 0 else int8.to_bytes(name)) for name in i])
def get_glyf_table(I:Interpreter):
    g = glyf()
    g.scaled_x, g.scaled_y = map(lambda x: [FU_to_px(I, x0) for x0 in x], [g.x, g.y]) # convert Funits to pixels
    g.fitted_x, g.fitted_y = map(list.copy, (g.scaled_x, g.scaled_y))
    g.touched = [False] * len(g.x)
    return g

class TestTTFOps(unittest.TestCase):
    def setUp(self):
        self.I = Interpreter()
        self.I.callstack = []
        self.I.fontsize = 12
        self.I.dpi = 144
        self.inputs = [
            0, 0, 1, 1, 2, 4,
            -2, 2, 9199213, -9199213,
            0, 1, 42, 9999991,
            -0, -1, -42, -9999991,
            0.5, -0.5, 12.52, -12.12,
            -2**33, 2**33,
            randint(-2**32, 2**32-1), randint(-2**32, 2**32-1), randint(-2**32, 2**32-1),
            random(), random(), random()
        ]
    def tearDown(self): del self.I

    def test_push(self):
        self.I.stack = [None] * 10
        self.I.sp = 0
        for i in self.inputs:
            push(self.I, int32.to_bytes(i))
            self.assertEqual(int32(self.I.stack[0]), int32(i))
            self.assertEqual(self.I.sp, 1)
            self.I.sp = 0
        with self.assertRaises(AssertionError): push(self.I, 0)
        with self.assertRaises(AssertionError): push(self.I, (int32.to_bytes(0),), int32.to_bytes(2))
        with self.assertRaises(AssertionError):
            self.I.sp = -1
            push(self.I, int32.to_bytes(10))
        with self.assertRaises(AssertionError):
            self.I.sp = 10
            push(self.I, int32.to_bytes(10))

        for p1, p2 in zip(self.inputs, self.inputs[1:]):
            p1, p2 = map(int32.to_bytes, (p1, p2))
            self.I.stack = [None] * 10
            self.I.sp = 0
            push(self.I, p1, p2)
            self.assertEqual(self.I.stack[0], p1)
            self.assertEqual(self.I.stack[1], p2)
            push(self.I, (p1,p2)) # tuple
            self.assertEqual(self.I.stack[2], p1)
            self.assertEqual(self.I.stack[3], p2)
            self.I.stack, self.I.sp = [], 0
        with self.assertRaises(AssertionError): push(self.I, int32.to_bytes(0))

    def test_pop(self):
        self.I.stack = [None] * 10
        p1 = int32.to_bytes(randint(0, 2**32))
        p2 = int32.to_bytes(randint(0, 2**32))
        self.I.stack[0], self.I.stack[1] = p1, p2
        self.I.sp = 2
        self.assertEqual(pop(self.I), p2)
        self.assertEqual(pop(self.I), p1)
        with self.assertRaises(AssertionError): pop(self.I)
        with self.assertRaises(AssertionError):
            self.I.sp = 11
            pop(self.I)
        self.I.sp = 3
        self.assertEqual(pop(self.I), None)
        self.I.stack, self.I.sp = [], 0
        with self.assertRaises(AssertionError): pop(self.I)

    def test_SVTCA(self):
        self.I.gs = {}
        self.I.run("SVTCA0")
        self.assertEqual([self.I.gs["projection_vector"], self.I.gs["freedom_vector"]], [vec2(0, 1), vec2(0, 1)])
        self.I.run("SVTCA1")
        self.assertEqual([self.I.gs["projection_vector"], self.I.gs["freedom_vector"]], [vec2(1, 0), vec2(1, 0)])
    
    def test_SPVTCA(self):
        self.I.gs = {}
        self.I.run("SPVTCA0") # Y
        self.assertEqual(self.I.gs["projection_vector"], vec2(0, 1))
        self.I.run("SPVTCA1") # Y
        self.assertEqual(self.I.gs["projection_vector"], vec2(1, 0))
    
    def test_SFVTCA(self):
        self.I.gs = {}
        self.I.run("SFVTCA0") # Y
        self.assertEqual(self.I.gs["freedom_vector"], vec2(0, 1))
        self.I.run("SFVTCA1") # x
        self.assertEqual(self.I.gs["freedom_vector"], vec2(1, 0))
    
    # def test_SPVTL(self): pass
    # def test_SFVTL(self): pass
    # def test_SPVFS(self): pass
    # def test_SFVFS(self): pass

    def test_GPV(self):
        pv = vec2(random(), random()).normalize()
        x, y = map(EF2Dot14, pv.components())
        self.I.gs = {"projection_vector": vec2(x,y)}
        self.I.stack, self.I.sp = [None]*2, 0
        self.I.run("GPV")
        self.assertEqual(EF2Dot14(self.I.stack[0]), x)
        self.assertEqual(EF2Dot14(self.I.stack[1]), y)

    def test_GFV(self):
        fv = vec2(random(), random()).normalize()
        x, y = map(EF2Dot14, fv.components())
        self.I.gs = {"freedom_vector": vec2(x,y)}
        self.I.stack, self.I.sp = [None]*2, 0
        self.I.run("GFV")
        self.assertEqual(EF2Dot14(self.I.stack[0]), x)
        self.assertEqual(EF2Dot14(self.I.stack[1]), y)

    def test_SFVTPV(self):
        v = vec2(random(), random()).normalize()
        self.I.gs = {"projection_vector": vec2(v.x, v.y)}
        self.I.run("SFVTPV")
        self.assertEqual(self.I.gs["projection_vector"], self.I.gs["freedom_vector"])
        self.I.gs["projection_vector"] += 2
        self.assertNotEqual(self.I.gs["projection_vector"], self.I.gs["freedom_vector"])

    # def test_ISECT(self): pass

    def test_SRP(self):
        for i in range(3):
            self.I.stack, self.I.sp = [None] * 5, 0
            self.I.gs = {f"rp{i}": None}
            self.I.run(["PUSHB0", 1, f"SRP{i}"])
            self.assertEqual(self.I.sp, 0)
            self.assertEqual(self.I.gs[f"rp{i}"], 1)
    def test_SZP(self):
        for i in range(3):
            self.I.stack, self.I.sp = [None] * 5, 0
            self.I.gs = {f"zp{i}": None}
            self.I.run(["PUSHB0", 1, f"SZP{i}"])
            self.assertEqual(self.I.sp, 0)
            self.assertEqual(self.I.gs[f"zp{i}"], 1)
    def test_SZPS(self):
        inputs = [0, 1, -1, 10, 0.5]
        self.I.gs = {"zp0": None, "zp1": None, "zp2": None}
        for i in inputs:
            i = uint32(i)
            self.I.stack, self.I.sp = [uint32.to_bytes(i), None], 1
            self.I.run("SZPS")
            self.assertEqual(self.I.gs, {"zp0": i, "zp1": i, "zp2": i})
            self.assertEqual(self.I.sp, 0)
    def test_SLOOP(self):
        self.I.stack, self.I.sp = [None] * 1, 0
        self.I.gs = {"loop": None}
        self.I.run(["PUSHB0", 5, "SLOOP"])
        self.assertEqual(self.I.sp, 0)
        self.assertEqual(self.I.gs["loop"] , 5)
    def test_SMD(self):
        self.I.stack, self.I.sp = [None] * 1, 0
        self.I.gs = {"minimum_distance": None}
        self.I.run(["PUSHB0", 5, "SMD"])
        self.assertEqual(self.I.sp, 0)
        self.assertEqual(self.I.gs["minimum_distance"] , 5 / 64) # F26Dot6
    def test_IF_ELSE_EIF(self):
        program = [
            "IF",
                "POP",
                "IF",
                    "POP",
                    "POP",
                "ELSE",
                    "POP",
                    "POP",
                    "POP",
                "EIF",
                "POP",
            "ELSE",
                "IF",
                    "POP",
                "ELSE",
                    "POP",
                    "POP",
                "EIF",
                "POP",
            "EIF",
            "POP"
        ]

        treasure, toPop, trueV, falseV = int32.to_bytes(-1), int32.to_bytes(5), int32.to_bytes(9000), int32.to_bytes(0) 
        stacks = [
            [treasure, toPop, toPop, toPop, toPop,         trueV,  toPop, trueV ],
            [treasure, toPop, toPop, toPop, toPop, toPop,  falseV, toPop, trueV ],
            [treasure, toPop, toPop, toPop,        trueV,                 falseV],
            [treasure, toPop, toPop, toPop, toPop, falseV,                falseV]
        ]
        for s in stacks:
            self.I.stack, self.I.sp = s, len(s)
            self.I.run(program)
            self.assertEqual(self.I.stack[self.I.sp - 1], treasure)

        # unterminated if statement
        program = [
            "IF",
                "POP",
                "IF",
                    "POP",
                    "POP",
                "ELSE",
                    "POP",
                    "POP",
                    "POP",
                "EIF",
                "POP",
            "ELSE",
                "IF",
                    "POP",
                "ELSE",
                    "POP",
                    "POP",
                "EIF",
                "POP"
        ]

        # unterminated if statement inside function definition
        program = [
            "FDEF",
                "IF",
                    "POP",
                    "IF",
                        "POP",
                        "POP",
                    "ELSE",
                        "POP",
                        "POP",
                        "POP",
                    "EIF",
                    "POP",
                "ELSE",
                    "IF",
                        "POP",
                    "ELSE",
                        "POP",
                        "POP",
                    "EIF",
                    "POP",
            "ENDF",
            "CALL"
        ]
        for s in stacks:
            self.I.maxp = maxp
            self.I.functions = [None]
            self.I.stack = s + [uint32.to_bytes(1), uint32.to_bytes(1)]
            self.I.sp = len(s)
            with self.assertRaises(AssertionError): self.I.run(program)

        # if IF and ELSE followed by values that are the same as relevant opcodes
        # test is somewhat reduced compared to the one in test_FDEF_ENDF_CALL_LOOPCALL because it builds on the same stuff
        # IF
        program = [
            "IF",
                "PUSHB0",
                "IF",
            "EIF"
        ]
        self.I.stack, self.I.sp = [uint32.to_bytes(1)], 1
        self.I.run(program)
        self.assertEqual(self.I.stack, [uint32.to_bytes(name_code["IF"])])
        # ELSE
        program  = [
            "IF",
            "ELSE",
                "PUSHB1",
                "ELSE",
                "EIF",
            "EIF"
        ]
        self.I.stack, self.I.sp = uint32_stack([0, None ]), 1
        self.I.run(program)
        self.assertEqual(self.I.stack, [uint32.to_bytes(name_code["ELSE"]), uint32.to_bytes(name_code["EIF"])])

    def test_JMPR(self):
        self.I.stack, self.I.sp = uint32_stack([5, 1, None]), 2
        with self.assertRaises(AssertionError): self.I.run(["PUSHB0", 3, "JMPR", "POP", "POP"])
        
        self.I.run(["PUSHB0", 3, "JMPR", "POP", "POP", "POP"])
        self.assertEqual(self.I.sp, 1)

        # do one loop using negative JMPR, then, to avoid an infinite loop use JROT to jump out
        self.I.stack, self.I.sp = [None] * 4, 0
        self.I.run(["PUSHW0", -1, -1, "PUSHB0", 1, "ADD", "DUP", "PUSHB0", 5, "SWAP", "JROT", "PUSHW0", -1, -11, "JMPR", "POP"])
        self.assertEqual(self.I.sp, 0)
        
    # def test_SCVTCI(self): pass
    def test_SSWCI(self):
        self.I.stack, self.I.sp = [None], 0
        self.I.gs = {"single_width_cut_in": None}
        self.I.run(["PUSHB0", 0xFF, "SSWCI"])
        self.assertEqual(self.I.gs["single_width_cut_in"], 4-1/64)
        self.assertEqual(self.I.sp, 0)

    def test_SSW(self):
        self.I.stack, self.I.sp = [None], 0
        self.I.fontsize, self.I.dpi = 12, 72
        self.I.head = head
        self.I.gs = {"single_width_value": None}
        self.I.run(["PUSHB0", 0xFF, "SSW"])
        self.assertEqual(self.I.gs["single_width_value"], F26Dot6(255 * 12 / 2048))
        self.assertEqual(self.I.sp, 0)

    def test_DUP(self):
        self.I.stack = int32_stack([-11456, None, None])
        self.I.sp = 1
        self.I.run("DUP")
        self.assertEqual(self.I.stack, int32_stack([-11456, -11456, None]))
        self.I.stack, self.I.sp = [int32.to_bytes(0)], 1
        with self.assertRaises(AssertionError): self.I.run("DUP")
        self.I.stack, self.I.sp = [int32.to_bytes(0), None], 1
        self.I.run("DUP")
        self.assertEqual(self.I.stack, int32_stack([0, 0]))
        self.assertEqual(self.I.sp, 2)
        self.I.stack, self.I.sp = [], 0
        with self.assertRaises(AssertionError): self.I.run("DUP")

    def test_CLEAR(self):
        self.I.stack, self.I.sp = [None] * 4, 0
        self.I.maxp = maxp
        self.I.run(["PUSHB3", 0xa2, 0x7b, 0x81, 0x04, "CLEAR"])
        self.assertEqual(self.I.stack, [None] * 4)
        self.assertEqual(self.I.sp, 0)

    def test_SWAP(self):
        self.I.stack = int32_stack([None, 5, 7])
        self.I.sp = 3
        self.I.run("SWAP")
        self.assertEqual(self.I.stack, int32_stack([None, 7, 5]))

    def test_DEPTH(self):
        self.I.stack = int32_stack([2, 2, None, None])
        self.I.sp = 2
        self.I.run("DEPTH")
        self.assertEqual(self.I.stack, int32_stack([2, 2, 2, None]))

    def test_CINDEX(self):
        valid_cases = [
            [3, -2, 5, 1, 2],
            [3, -2, 5, 1, 1],
            [3, -2, 5, 1, 4]
        ]
        invalid_cases = [
            [3, -2, 5, 1, 5], # index too big
            [3, -2, 5, 1, 0], # index 0
            [3, -2, 5, 1, -2] # negative index
        ]
        for c in valid_cases: 
            self.I.stack, self.I.sp = int32_stack(c), len(c)    
            self.I.run("CINDEX")
            self.assertEqual(self.I.stack, int32_stack(c[:4] + [c[-c[-1] - 1]]))
            self.assertEqual(self.I.sp, len(c))
        for c in invalid_cases: 
            self.I.stack, self.I.sp = int32_stack(c), len(c)
            with self.assertRaises(AssertionError): self.I.run("CINDEX")

    def test_MINDEX(self):
        valid_cases = [
            [3, -2, 5, 1, 2],
            [3, -2, 5, 1, 1],
            [3, -2, 5, 1, 4]
        ]
        results = [
            [3, -2, 1, 5, 2],
            [3, -2, 5, 1, 1],
            [-2, 5, 1, 3, 4]
        ]
        invalid_cases = [
            [3, -2, 5, 1, 5], # index too big
            [3, -2, 5, 1, 0], # index 0
            [3, -2, 5, 1, -2] # negative index
        ]
        for c, r in zip(valid_cases, results):
            self.I.stack, self.I.sp = int32_stack(c), len(c)
            self.I.run("MINDEX")
            self.assertEqual(self.I.stack, int32_stack(r))
            self.assertEqual(self.I.sp, len(c) - 1)
        for c in invalid_cases: 
            self.I.stack, self.I.sp = int32_stack(c), len(c)
            with self.assertRaises(AssertionError): self.I.run("MINDEX")

    # def test_ALIGNPTS(self): pass
    # def test_UTP(self): pass

    def test_FDEF_ENDF_CALL_LOOPCALL(self):
        self.I.maxp = maxp()
        program = [
            "FDEF",
                "POP",
            "ENDF",
            "FDEF",
                "POP",
                "POP",
            "ENDF",
            "CALL",
            "LOOPCALL"
        ]
        self.I.functions = [None, None]
        v1, v5, v_5, v3, v0, v4 = map(int32.to_bytes, (1, 5, -5, 3, 0, 4))
        self.I.stack, self.I.sp = [v3, v3, v3, v3, v4, v0, v3, v0, v1, v0], 10
        self.I.run(program)
        self.assertEqual(self.I.functions[0]["ip"], 0)
        self.assertEqual(self.I.functions[1]["ip"], 3)
        self.assertEqual(self.I.sp, 0)

        self.I.stack, self.I.sp = [v3, v3, v3, v3, v0, v0, v3, v0, v1, v0], 10
        self.I.run(program)
        self.assertEqual(self.I.sp, 4)

        self.I.stack, self.I.sp = [v3, v3, v3, v3, v4, v0, v3, v0, v1], 9
        with self.assertRaises(AssertionError): self.I.run(program)

        self.I.stack, self.I.sp = [v3, v3, v3, v3, v5, v0, v3, v0, v0], 9
        with self.assertRaises(AssertionError): self.I.run(program)

        self.I.stack, self.I.sp = [v3, v3, v3, v3, v5, v0, v3, v1, v0], 9
        with self.assertRaises(AssertionError): self.I.run(program)

        self.I.stack, self.I.sp = [v3, v3, v3, v3, v_5, v0, v3, v0, v0], 9
        with self.assertRaises(AssertionError): self.I.run(program)

        program2 = ["CALL", "POP"]
        self.I.stack, self.I.sp = [v3, v3, v0, v3, v1, v0, v3, v0, v1, v0], 10
        self.I.run(program)
        self.assertEqual(self.I.sp, 3)
        self.I.run(program2)
        self.assertEqual(self.I.sp, 0)

        self.I.stack, self.I.sp = [v3, v3, v0, v3, v1, v1, v3, v0, v1, v0], 10
        self.I.run(program)
        self.assertEqual(self.I.sp, 2)

        # test differentiation between op and value in FDEF
        for n in range(8):
            # test with PUSHB
            n_true = n+1
            program = ["FDEF", f"PUSHB{n}"] + ["FDEF"] * n_true + ["ENDF"] 
            self.I.stack, self.I.sp = uint32_stack([0]), 1
            self.I.run(program)
            # test with PUSHW
            program = ["FDEF", f"PUSHW{n}"] + ["FDEF"] * n_true * 2 + ["ENDF"]
            self.I.stack, self.I.sp = uint32_stack([0]), 1
            self.I.run(program)
        for i in range(4,20):
            # test with NPUSHB
            program = ["FDEF", "NPUSHB", uint8(i)] + ["FDEF"] * i + ["ENDF"]
            self.I.stack, self.I.sp = uint32_stack([0]), 1
            self.I.run(program)
            # test with NPUSHW
            program = ["FDEF", "NPUSHW", uint8(i)] + ["FDEF"] * i * 2 + ["ENDF"]
            self.I.stack, self.I.sp = uint32_stack([0]), 1
            self.I.run(program)

    def test_MDAP(self):
        self.I.gs = {"zp0": 1,"rp0": None, "rp1": None, "freedom_vector": None, "projection_vector": None, "minimum_distance": 0}
        p = vec2(-2.711, 1.234)
        self.I.g = glyf()
        pvs = [vec2(-1, 0), vec2(1, 0), vec2(0.866, 0.5), vec2(-2,-3)]
        fvs = [vec2(-1, 0), vec2(1, 0), vec2(1, 0), vec2(0, 1)]
        self.I.run("RTG") # set round state
        for pv, fv in zip(pvs, fvs):
            self.I.g.scaled_x = [p.x]
            self.I.g.scaled_y = [p.y]
            self.I.g.fitted_x, self.I.g.fitted_y = self.I.g.scaled_x, self.I.g.scaled_y
            self.I.g.touched = [False]
            self.I.gs["projection_vector"], self.I.gs["freedom_vector"] = pv.normalize(), fv.normalize()
            self.I.stack, self.I.sp = uint32_stack([0, 0]), 2
            self.I.run("MDAP1")
            self.assertEqual(self.I.sp, 1)
            coord = orthogonal_projection(p, self.I.gs["projection_vector"])
            ncoord = (coord + 0.5) // 1 # rounding
            ratio = orthogonal_projection(self.I.gs["freedom_vector"], self.I.gs["projection_vector"])
            result = p + (ncoord - coord) / ratio * self.I.gs["freedom_vector"]
            self.assertAlmostEqual(self.I.g.fitted_x[0], result.x, msg=f"{pv=}, {fv=}")
            self.assertAlmostEqual(self.I.g.fitted_y[0], result.y)
            self.assertEqual(self.I.gs["rp0"], 0)
            self.assertEqual(self.I.gs["rp1"], 0)

            self.I.gs["rp1"] = self.I.gs["rp0"] = None
            self.I.twilight = [p]
            self.I.run("MDAP0") # no change
            self.assertEqual(self.I.sp, 0)
            self.assertEqual(self.I.twilight[0].x, p.x)
            self.assertEqual(self.I.twilight[0].y, p.y)
            self.assertEqual(self.I.gs["rp0"], 0)
            self.assertEqual(self.I.gs["rp1"], 0)
            self.assertEqual(self.I.g.touched[0], True)
            

    def test_IUP(self):
        self.I.g = glyf()

        # IUP1 with point between other ones
        self.I.g.scaled_x = [0, 2, 5]
        self.I.g.scaled_y = [0, 0, 0]
        self.I.g.fitted_x = [1, 2, 11]
        self.I.g.fitted_y = [0, 0, 0]
        self.I.g.touched = [True, False, True]
        self.I.g.endPtsContours = [2]
        self.I.gs = {"zp2": 1}
        self.I.run("IUP1")
        self.assertEqual(self.I.g.touched, [True, False, True]) # unchanged
        self.assertEqual(self.I.g.fitted_x, [1, 5, 11])

        # wrong zone
        self.I.gs = {"zp2": 0}
        with self.assertRaises(AssertionError): self.I.run("IUP1")

        # IUP0 with point between other ones
        self.I.g.scaled_x = [0, 0, 0]
        self.I.g.scaled_y = [0, 2, 5]
        self.I.g.fitted_x = [0, 0, 0]
        self.I.g.fitted_y = [0, 2, 10]
        self.I.g.touched = [True, False, True]
        self.I.g.endPtsContours = [2]
        self.I.gs = {"zp2": 1}
        self.I.run("IUP0")
        self.assertEqual(self.I.g.touched, [True, False, True]) # unchanged
        self.assertEqual(self.I.g.fitted_y, [0, 4, 10])

        # IUP0 with three points, the one to move at the beginning
        self.I.g.scaled_x = [0, 0, 0]
        self.I.g.scaled_y = [2, 5, 0]
        self.I.g.fitted_x = [0, 0, 0]
        self.I.g.fitted_y = [2, 10, 0]
        self.I.g.touched = [False, True, True]
        self.I.g.endPtsContours = [2]
        self.I.gs = {"zp2": 1}
        self.I.run("IUP0")
        self.assertEqual(self.I.g.touched, [False, True, True]) # unchanged
        self.assertEqual(self.I.g.fitted_y, [4, 10, 0])

        # IUP0 with three points, the one to move at the end
        self.I.g.scaled_x = [0, 0, 0]
        self.I.g.scaled_y = [5, 0, 2]
        self.I.g.fitted_x = [0, 0, 0]
        self.I.g.fitted_y = [10, 0, 2]
        self.I.g.touched = [True, True, False]
        self.I.g.endPtsContours = [2]
        self.I.gs = {"zp2": 1}
        self.I.run("IUP0")
        self.assertEqual(self.I.g.touched, [True, True, False]) # unchanged
        self.assertEqual(self.I.g.fitted_y, [10, 0, 4])

        # IUP0 with two points (= point to move not between)
        self.I.g.scaled_x = [0, 0]
        self.I.g.scaled_y = [0, 2]
        self.I.g.fitted_x = [0, 0]
        self.I.g.fitted_y = [0, 4]
        self.I.g.touched = [False, True]
        self.I.g.endPtsContours = [1]
        self.I.gs = {"zp2": 1}
        self.I.run("IUP0")
        self.assertEqual(self.I.g.touched, [False, True]) # unchanged
        self.assertEqual(self.I.g.fitted_y, [2, 4])

        # IUP1 with many points, between and not between
        self.I.g.scaled_x = [0, 5,  7, 3, -5]
        self.I.g.scaled_y = [0, 0,  8, 5,  1]
        self.I.g.fitted_x = [0, 5, 14, 4, -5]
        self.I.g.fitted_y = [1, 0,  5, 5,  1]
        self.I.g.touched = [True, False, True, True, False]
        self.I.g.endPtsContours = [4]
        self.I.gs = {"zp2": 1}
        self.I.run("IUP1")
        self.assertEqual(self.I.g.touched, [True, False, True, True, False]) # unchanged
        self.assertEqual(self.I.g.fitted_x, [0, 10, 14, 4, -5])
        self.assertEqual(self.I.g.fitted_y, [1,  0,  5, 5,  1]) # unchanged

        # IUP0 with nothing to move
        self.I.g.scaled_x = [0, 0, 0, 0]
        self.I.g.scaled_y = [0, 2, 5, 1]
        self.I.g.fitted_x = [0, 0, 0, 0,]
        self.I.g.fitted_y = [0, 2, 10, 2]
        self.I.g.touched = [False, False, True, True]
        self.I.g.endPtsContours = [2]
        self.I.gs = {"zp2": 1}
        self.I.run("IUP0")
        self.assertEqual(self.I.g.touched, [False, False, True, True]) # unchanged
        self.assertEqual(self.I.g.fitted_y, [0, 2, 10, 2]) # unchanged

        # IUP0 with multiple contours between and not between
        self.I.g.scaled_x = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.I.g.scaled_y = [0, 2, 5, 7, 1, 2, 0, 2, 5]
        self.I.g.fitted_x = [0, 0, 0, 0, 1, 0, 0, 0, 0]
        self.I.g.fitted_y = [1, 2, 11, 7, 1, 3, 1, 2, 11]
        self.I.g.touched = [True, False, True, False, True, True, True, False, True]
        self.I.g.endPtsContours = [2, 5, 8]
        self.I.gs = {"zp2": 1}
        self.I.run("IUP0")
        self.assertEqual(self.I.g.touched, [True, False, True, False, True, True, True, False, True]) # unchanged
        self.assertEqual(self.I.g.fitted_y, [1, 5, 11, 8, 1, 3, 1, 5, 11])

    # def test_SHP(self): pass
    # def test_SHC(self): pass
    # def test_SHZ(self): pass
    def test_SHPIX(self):
        fvs = [vec2(1,1), vec2(0.5, 5), vec2(-1, 0), vec2(-19, 0.1)]
        ds = [F26Dot6(1.241), 0, -3, F26Dot6(0.7)]
        for fv, d in zip(fvs, ds):
            fv = fv.normalize()
            self.I.gs = {"zp2": 1, "loop": 3, "freedom_vector": fv, "projection_vector": vec2(100, 100)} # pv only used for assert
            shift = d * fv
            self.I.g = glyf()
            self.I.g.scaled_x = [4, 1, 0.4]
            self.I.g.scaled_y = [-1, 4151, -0.1235]
            self.I.g.fitted_x, self.I.g.fitted_y = self.I.g.scaled_x.copy(), self.I.g.scaled_y.copy()
            self.I.g.touched = [False] * 3
            self.I.stack, self.I.sp = int32_stack([0, 1, 2, int32(F26Dot6.to_bytes(d))]), 4
            self.assertEqual(F26Dot6(self.I.stack[-1]), d) 
            self.I.run("SHPIX")
            self.assertEqual(self.I.gs["loop"], 1)
            self.assertEqual(self.I.sp, 0)
            self.assertEqual(self.I.g.fitted_x, [p + shift.x for p in self.I.g.scaled_x])
            self.assertEqual(self.I.g.fitted_y, [p + shift.y for p in self.I.g.scaled_y])
            self.assertEqual(self.I.g.touched, [True] * 3)
        self.I.gs = {"zp2": 1, "loop": 3, "freedom_vector": vec2(1,1), "projection_vector": vec2(-1, -1)}
        with self.assertRaises(AssertionError): self.I.run("SHPIX")

    # def test_IP(self): pass
    # def test_MSIRP(self): pass
    def test_ALIGNRP(self):
        self.I.gs = {"rp0": 3, "loop": 3, "zp0": 0, "zp1": 0, "freedom_vector": vec2(1,0), "projection_vector": vec2(1,0)}
        self.I.twilight = [vec2(0.2, 0.5), vec2(5, 8), vec2(-500, 8),vec2(1,2)]
        self.I.stack, self.I.sp = uint32_stack([0, 1, 2]), 3
        self.I.run("ALIGNRP")
        self.assertTrue(all([v.x == 1 for v in self.I.twilight]))
        self.assertEqual(self.I.gs["loop"], 1)

        self.I.gs = {"rp0": 3, "loop": 3, "zp0": 0, "zp1": 0, "freedom_vector": vec2(0,-1), "projection_vector": vec2(0,1)}
        self.I.twilight = [vec2(0.2, 0.5), vec2(5, 8), vec2(-500, 8),vec2(1,2)]
        self.I.stack, self.I.sp = uint32_stack([0, 1, 2]), 3
        self.I.run("ALIGNRP")
        self.assertTrue(all([v.y == 2 for v in self.I.twilight]))
        self.assertEqual(self.I.gs["loop"], 1)

        self.I.gs = {"rp0": 3, "loop": 3, "zp0": 0, "zp1": 0, "freedom_vector": vec2(-1,-1), "projection_vector": vec2(0,1)}
        self.I.twilight = [vec2(0.2, 0.5), vec2(5, 8), vec2(-500, 8),vec2(1,2)]
        self.I.stack, self.I.sp = uint32_stack([0, 1, 2]), 3
        self.I.run("ALIGNRP")
        self.assertTrue(all([v.y == 2 for v in self.I.twilight]))
        self.assertEqual(self.I.gs["loop"], 1)

    def test_MIAP(self): # testing without rounding or control value cut in
        # pv and fv in x direction
        self.I.gs = {"projection_vector": vec2(1,0), "freedom_vector": vec2(1,0), "zp0": 0, "rp0": None, "rp1": None}
        self.I.cvt = [0, 2]
        self.I.stack, self.I.sp = int32_stack([0, 1]), 2
        self.I.twilight = [vec2(1,1)]
        self.I.run("MIAP0")
        self.assertEqual(self.I.twilight[0], vec2(2,1))

        # pv and fv in y direction
        self.I.gs = {"projection_vector": vec2(0,1), "freedom_vector": vec2(0,1), "zp0": 0, "rp0": None, "rp1": None}
        self.I.cvt = [0, 3]
        self.I.stack, self.I.sp = int32_stack([0, 1]), 2
        self.I.twilight = [vec2(1,1)]
        self.I.run("MIAP0")
        self.assertEqual(self.I.twilight[0], vec2(1,3))

        # pv in x in directionand fv at 45 degrees 
        self.I.gs = {"projection_vector": vec2(1,0), "freedom_vector": vec2(1,1).normalize(), "zp0": 0, "rp0": None, "rp1": None}
        self.I.cvt = [0, 3]
        self.I.stack, self.I.sp = int32_stack([0, 1]), 2
        self.I.twilight = [vec2(1,1)]
        self.I.run("MIAP0")
        self.assertEqual(self.I.twilight[0], vec2(3,3))

    def test_NPUSHB(self):
        self.I.stack, self.I.sp = [None, None, None], 0
        self.I.run(["NPUSHB", 0x2, 0x28, 0xf2])
        self.assertEqual(self.I.stack, [b'\x00\x00\x00\x28', b'\x00\x00\x00\xf2', None])
        self.assertEqual(self.I.sp, 2)

        self.I.stack, self.I.sp = [None, None, None], 0
        self.I.run(["NPUSHB", 0])
        self.assertEqual(self.I.stack, [None, None, None])
        self.assertEqual(self.I.sp, 0)

        self.I.stack, self.I.sp = [None, None, None], 0
        with self.assertRaises(AssertionError): self.I.run(["NPUSHB", 0x3, 0x28, 0xf2])

    def test_NPUSHW(self):
        self.I.stack, self.I.sp = [None, None, None], 0
        self.I.run(["NPUSHW", 0x2, 0x28, 0xf2, 0x5d, 0xa])
        self.assertEqual(self.I.stack, int32_stack([0x28f2, 0x5d0a, None]))
        self.assertEqual(self.I.sp, 2)

        self.I.stack, self.I.sp = [None, None, None], 0
        self.I.run(["NPUSHW", 0x0])
        self.assertEqual(self.I.stack, [None, None, None])
        self.assertEqual(self.I.sp, 0)

        self.I.stack, self.I.sp = [None, None, None], 0
        with self.assertRaises(AssertionError): self.I.run(["NPUSHW", 0x2, 0x28, 0xf2, 0x5d])

    def test_WS_RS(self):
        # WS
        self.I.stack, self.I.sp = uint32_stack([6, 0, 0, 5]), 4
        self.I.maxp, self.I.storage = maxp(), [None]
        self.I.run("WS")
        self.assertEqual(self.I.storage[0], uint32(5))
        self.assertEqual(self.I.sp, 2)
        with self.assertRaises(AssertionError): self.I.run("WS")

        # RS
        self.I.stack, self.I.sp = uint32_stack([6, 0, 0, 5]), 4
        self.I.maxp, self.I.storage = maxp(), [None]
        self.I.run("WS")
        self.assertEqual(self.I.storage[0], uint32(5))
        self.assertEqual(self.I.sp, 2)
        self.I.run("RS")
        self.assertEqual(self.I.stack[:2], uint32_stack([6, self.I.storage[0]]))
        self.assertEqual(self.I.sp, 2)

        self.I.storage = [None, uint32(90)]
        self.I.stack, self.I.sp = uint32_stack([4, 1, None]), 2
        self.I.run("RS")
        self.assertEqual(self.I.stack, uint32_stack([4, 90, None]))
        self.assertEqual(self.I.sp, 2)
        with self.assertRaises(AssertionError): self.I.run("RS")

    def test_WCVTP(self):
        self.I.stack, self.I.sp = [uint32.to_bytes(5), F26Dot6.to_bytes(-1526.5), uint32.to_bytes(0), F26Dot6.to_bytes(15.125)], 4
        self.I.cvt = [None] * 2
        self.I.run("WCVTP")
        self.assertEqual(self.I.cvt[0], F26Dot6(15.125))
        self.assertEqual(self.I.sp, 2)
        # RCVT
        push(self.I, uint32.to_bytes(0))
        self.I.run("RCVT")
        self.assertEqual(self.I.stack[3], F26Dot6.to_bytes(15.125))
        self.assertEqual(self.I.sp, 3)
        with self.assertRaises(AssertionError): self.I.run("WCVTP")
        
    def test_RCVT(self):
        self.I.cvt = [F26Dot6(i) for i in self.inputs]
        for i in range(len(self.I.cvt)):
            self.I.stack, self.I.sp = [uint32.to_bytes(i), None], 1
            self.I.run("RCVT")
            self.assertEqual(self.I.stack, [F26Dot6.to_bytes(self.I.cvt[i]), None])
            self.assertEqual(self.I.sp, 1)
        for i in self.inputs:
            v = int32(i)
            self.I.stack, self.I.sp = [int32.to_bytes(v), None], 1
            if len(self.I.cvt) > v >= 0:
                self.I.run("RCVT")
                self.assertEqual(self.I.stack, [F26Dot6.to_bytes(self.I.cvt[v]), None])
                self.assertEqual(self.I.sp, 1)
            else:
                with self.assertRaises(AssertionError): self.I.run("RCVT")

    def test_GC(self):
        self.I.head = head
        self.I.g = get_glyf_table(self.I)
        self.I.twilight = twilight_points
        def test():
            p = self.I.get_point(i, "zp2", "original" if a else "fitted")
            pjs = [vec2(0,1), vec2(1,0), vec2(-p.x, -p.y).normalize() if p.x != 0 or p.y != 0 else vec2(1,0)]
            results = [p.y, p.x, -p.mag() if p.x != 0 or p.y != 0 else 0]
            for pj, res in zip(pjs, results):
                self.I.stack, self.I.sp = [uint32.to_bytes(i)], 1
                self.I.gs["projection_vector"] = pj
                self.I.run(f"GC{a}")
                self.assertAlmostEqual(F26Dot6(self.I.stack[0]), F26Dot6(res))
        for a in [0, 1]:
            self.I.gs = {"zp2": 1, "projection_vector": vec2(1, 0)}  # test glyph zone
            for i in range(len(self.I.g.x)): test()
            self.I.gs["zp2"] = 0 # test twilight zone
            for i in range(len(self.I.twilight)): test()

    def test_SCFS(self): pass
    def test_MD(self):
        self.I.head = head
        self.I.g = get_glyf_table(self.I)
        self.I.twilight = twilight_points
        def test():
            p1 = self.I.get_point(i, "zp1", "original" if a else "fitted")
            p2 = self.I.get_point(i+1, "zp0", "original")
            self.I.stack, self.I.sp = [uint32.to_bytes(i+1), uint32.to_bytes(i)], 2
            self.I.run(f"MD{a}")
            res = orthogonal_projection(p2, self.I.gs["projection_vector"]) - orthogonal_projection(p1, self.I.gs["projection_vector"])
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(res))
        
        for a in [0,1]:
            self.I.gs = {"zp0": 1, "zp1": 1, "projection_vector": vec2(1, 0)} # test glyph zone
            for i in range(0, len(self.I.g.x)-1, 2): test()
            self.I.gs["zp0"] = self.I.gs["zp1"] = 0 # test twilight zone
            for i in range(0, len(self.I.twilight)-1, 2): test()

    def test_MPPEM(self):
        self.I.fontsize, self.I.dpi = 12, 144
        self.I.stack, self.I.sp = [None], 0
        self.I.run("MPPEM")
        self.assertEqual(uint32(self.I.stack[0]), int(12 * 144 / 72))
    # def test_MPS(self): pass
    # def test_FLIPON(self): pass
    # def test_FLIPOFF(self): pass
    def test_DEBUG(self): pass
    def test_LT(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(int32.to_bytes, (i1, i2))), 2
            self.I.run("LT")
            self.assertEqual(self.I.stack[0], int32.to_bytes(int(int32(i1) < int32(i2))))
    def test_LTEQ(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(int32.to_bytes, (i1, i2))), 2
            self.I.run("LTEQ")
            self.assertEqual(self.I.stack[0], int32.to_bytes(int(int32(i1) <= int32(i2))))
    def test_GT(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(int32.to_bytes, (i1, i2))), 2
            self.I.run("GT")
            self.assertEqual(self.I.stack[0], int32.to_bytes(int(int32(i1) > int32(i2))))
    def test_GTEQ(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(int32.to_bytes, (i1, i2))), 2
            self.I.run("GTEQ")
            self.assertEqual(self.I.stack[0], int32.to_bytes(int(int32(i1) >= int32(i2))))
    def test_EQ(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(int32.to_bytes, (i1, i2))), 2
            self.I.run("EQ")
            self.assertEqual(self.I.stack[0], int32.to_bytes(int(int32(i1) == int32(i2))))
    def test_NEQ(self): 
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(int32.to_bytes, (i1, i2))), 2
            self.I.run("NEQ")
            self.assertEqual(self.I.stack[0], int32.to_bytes(int(int32(i1) != int32(i2))))
    # def test_ODD(self): pass
    # def test_EVEN(self): pass
    def test_AND(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(uint32.to_bytes, (i1, i2))), 2
            self.I.run("AND")
            self.assertEqual(self.I.stack[0], uint32.to_bytes(int(uint32(i1) != 0 and uint32(i2) != 0)))
            self.assertEqual(self.I.sp, 1)

    def test_OR(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(uint32.to_bytes, (i1, i2))), 2
            self.I.run("OR")
            self.assertEqual(self.I.stack[0], uint32.to_bytes(int(uint32(i1) != 0 or uint32(i2) != 0)))
            self.assertEqual(self.I.sp, 1)

    def test_NOT(self):
        for i in self.inputs:
            self.I.stack, self.I.sp = [uint32.to_bytes(i)], 1
            self.I.run("NOT")
            self.assertEqual(self.I.stack[0], uint32.to_bytes(int(not uint32(i))))
            self.assertEqual(self.I.sp, 1)

    # def test_DELTAP1(self): pass
    def test_SDB(self):
        inputs = [2**32-1, 0, 1, -1231, 123.45]
        for i in inputs:
            self.I.gs = {"delta_base": 9}
            self.I.stack, self.I.sp = [uint32(i), None], 1
            self.I.run("SDB")
            self.assertEqual(self.I.gs["delta_base"], uint32(i))
    def test_SDS(self):
        inputs = [2**32-1, 0, 1, -1231, 123.45]
        for i in inputs:
            self.I.gs = {"delta_shift": 3}
            self.I.stack, self.I.sp = [uint32(i), None], 1
            self.I.run("SDS")
            self.assertEqual(self.I.gs["delta_shift"], uint32(i))

    # NOTE: F26Dot6's low precision easily causes errors in arithemtic even at two decimal points
    def test_ADD(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack = list(map(lambda x: F26Dot6.to_bytes(x), (i1, i2)))
            self.I.sp = 2
            self.I.run("ADD")
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(sum((F26Dot6(i1), F26Dot6(i2)))))
            self.assertEqual(self.I.sp, 1)

    def test_SUB(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack = list(map(lambda x: F26Dot6.to_bytes(x), (i1, i2)))
            self.I.sp = 2
            self.I.run("SUB")
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(F26Dot6(i1) - F26Dot6(i2)))
            self.assertEqual(self.I.sp, 1)

    def test_DIV(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack = list(map(lambda x: F26Dot6.to_bytes(x), (i1, i2)))
            self.I.sp = 2
            self.I.run("DIV")
            if F26Dot6(i2) == 0:
                self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6((-1 if i1 < 0 else 1) * 0xFFFFFFFF))
            else:
                self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(F26Dot6(i1) / F26Dot6(i2)))
        self.I.stack, self.I.sp = [F26Dot6.to_bytes(1), F26Dot6.to_bytes(0)], 2
        self.I.run("DIV")
        self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(0xFFFFFFFF))

    def test_MUL(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack = list(map(lambda x: F26Dot6.to_bytes(x), (i1, i2)))
            self.I.sp = 2
            self.I.run("MUL")
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(F26Dot6(i1) * F26Dot6(i2)))

    def test_ABS(self):
        for i in self.inputs:
            self.I.stack = [F26Dot6.to_bytes(i)]
            self.I.sp = 1
            self.I.run("ABS")
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(abs(i)))

    def test_NEG(self):
        for i in self.inputs:
            self.I.stack = [F26Dot6.to_bytes(i)]
            self.I.sp = 1
            self.I.run("NEG")
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(-F26Dot6(i)))

    def test_FLOOR(self):
        for i in self.inputs:
            self.I.stack = [F26Dot6.to_bytes(i)]
            self.I.sp = 1
            self.I.run("FLOOR")
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(math.floor(F26Dot6(i))))

    def test_CEILING(self):
        for i in self.inputs:
            self.I.stack = [F26Dot6.to_bytes(i)]
            self.I.sp = 1
            self.I.run("CEILING")
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(math.ceil(F26Dot6(i))))

    def test_ROUND(self):
        rounding = ["RTHG", "RTG", "RTDG", "RDTG", "RUTG", "ROFF"] #, "SROUND", "S45ROUND"]
        self.I.gs = {"round_state": False}
        for i in self.inputs:
            i = F26Dot6(i)
            for r in rounding:
                self.I.run(r)
                self.I.stack, self.I.sp = [F26Dot6.to_bytes(i)], 1
                self.I.run("ROUND0")
                match r:
                    case "RTHG": res = (i + 1) // 1 - 0.5
                    case "RTG": res = (i + 0.5) // 1
                    case "RTDG": res = (i + 0.25) // 0.5 * 0.5
                    case "RDTG": res = math.floor(i)
                    case "RUTG": res = math.ceil(i)
                    case "ROFF": res = i
                self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(res))
                self.assertEqual(self.I.sp, 1)

    # def test_NROUND(self): pass
    def test_WCVTF(self):
        for i in self.inputs:
            self.I.stack, self.I.sp = uint32_stack([5, 1526, 0]) + [int32.to_bytes(i)], 4
            self.I.fontsize = 12
            self.I.dpi = 144
            self.I.head = head
            self.I.cvt = [None] * 2
            self.I.run("WCVTF")
            self.assertEqual(self.I.cvt[0], F26Dot6(int32(i) * ((self.I.fontsize * self.I.dpi) / (72*self.I.head.unitsPerEM))))
            self.assertEqual(self.I.sp, 2)
            with self.assertRaises(AssertionError): self.I.run("WCVTP")

    # def test_DELTAP2(self): pass
    # def test_DELTAP3(self): pass
    # def test_DELTAC1(self): pass
    # def test_DELTAC2(self): pass
    # def test_DELTAC3(self): pass
    # def test_SROUND(self): pass
    # def test_S45ROUND(self): pass
    def test_JROT(self):
        self.I.stack, self.I.sp = uint32_stack([5, 2, 1]), 3
        self.I.run(["JROT", "POP", "POP"])
        self.assertEqual(self.I.sp, 0)
        self.I.stack, self.I.sp = int32_stack([5, 0, 5, -1, 1, 3]), 6
        self.I.run(["POP", "JROT"])
        self.assertEqual(self.I.sp, 0)
    # def test_JROF(self): pass
    # def test_SANGW(self): pass
    # def test_AA(self): pass
    # def test_FLIPPT(self): pass
    # def test_FLIPRGON(self): pass
    # def test_FLIPRGOFF(self): pass
    # def test_SCANCTRL(self): pass TODO
    # def test_SDPVTL(self): pass
    def test_GETINFO(self):
        selector = int32.to_bytes(-1) # all 1s in binary
        self.I.stack, self.I.sp = [selector], 1
        self.I.run("GETINFO")
        self.assertEqual(self.I.stack, [int32.to_bytes(2)]) # only engine version is returned as 2, rest is 0
        self.assertEqual(self.I.sp, 1)
        self.I.stack, self.I.sp = [uint32.to_bytes(0)], 1 # get nothing
        self.I.run("GETINFO")
        self.assertEqual(self.I.stack, [uint32.to_bytes(0)])
        self.I.stack, self.I.sp = [uint32.to_bytes(1)], 1 # get only engine version
        self.I.run("GETINFO")
        self.assertEqual(self.I.stack, [uint32.to_bytes(2)])

    # def test_IDEF(self): pass

    def test_ROLL(self):
        valid_cases = [
            [2, None, 0, 0xff, 0xd3, None],
            [2, 0, 0xff, 0xd3, None],
            [2, 0, 0xff, None]
        ]
        results = [
            [2, None, 0xff, 0xd3, 0, None],
            [2, 0xff, 0xd3, 0, None],
            [0, 0xff, 2, None]
        ]
        invalid_case = [2,0]
        for c,r in zip(valid_cases, results):
            self.I.stack, self.I.sp = int32_stack(c), len(c) - 1
            self.I.run("ROLL")
            self.assertEqual(self.I.stack, int32_stack(r))
            self.assertEqual(self.I.sp, len(c) - 1)
        self.I.stack, self.I.sp = int32_stack(invalid_case), len(invalid_case)
        with self.assertRaises(AssertionError): self.I.run("ROLL")

    # def test_MAX(self): pass
    # def test_MIN(self): pass
    # def test_SCANTYPE(self): pass TODO
    # def test_INSTCTRL(self): pass TODO
    def test_PUSHB(self):
        self.I.stack, self.I.sp = [None, None, None], 0
        self.I.run(["PUSHB0", 0x2])
        self.assertEqual(self.I.stack, int32_stack([2, None, None]))
        self.assertEqual(self.I.sp, 1)

        self.I.stack, self.I.sp = [None, None, None], 0
        self.I.run(["PUSHB2", 0, 0xff, 0xd3])
        self.assertEqual(self.I.stack, int32_stack([0, 0xff, 0xd3]))
        self.assertEqual(self.I.sp, 3)

        self.I.stack, self.I.sp = [None, None, None], 0
        with self.assertRaises(AssertionError): self.I.run(["PUSHB3", 0x3, 0x28, 0xf2])

    def test_PUSHW(self):
        self.I.stack, self.I.sp = [None, None, None], 0
        self.I.run(["PUSHW0", 0x02, 0x42])
        self.assertEqual(self.I.stack, int32_stack([0x242, None, None]))
        self.assertEqual(self.I.sp, 1)

        self.I.stack, self.I.sp = [None, None, None], 0
        self.I.run(["PUSHW1", 0x00, 0xff, 0xd3, 0x23])
        self.assertEqual(self.I.stack, uint32_stack([0xff, 0xffffd323, None])) # second word is negative because 0xd3 leads with a 1 in binary.
        self.assertEqual(self.I.sp, 2)

        self.I.stack, self.I.sp = [None, None, None], 0
        with self.assertRaises(AssertionError): self.I.run(["PUSHW2", 0x03, 0x28, 0xf2, 0xff, 0xb0])

    # def test_MDRP(self): pass
    # def test_MIRP(self): pass

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
            I = Interpreter(fontfile=f)
            with open(f, "rb") as f: self.assertTrue(all(I.checksum(f.read())))


if __name__ == "__main__": unittest.main()