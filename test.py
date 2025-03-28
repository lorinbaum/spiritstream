import unittest, math
from ttf import *
from random import random, randint
from typing import List, Union
from table import glyf

class maxp:
    maxFunctionDefs = 2
    maxStorage = 2

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
def to_program(I:Interpreter, i:List[str]): return b''.join([uint8.to_bytes(I.opcodes[name] if type(name) == str else name) for name in i])
def get_glyf_table(I:Interpreter):
    g = glyf()
    g.scaled_x, g.scaled_y = map(lambda x: [I.FU_to_px(x0) for x0 in x], [g.x, g.y]) # convert Funits to pixels
    g.fitted_x, g.fitted_y = map(list.copy, (g.scaled_x, g.scaled_y))
    return g

class TestTTFOps(unittest.TestCase):
    def setUp(self):
        self.I = Interpreter()
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
        p1 = int.to_bytes(randint(-2**31, 2**31), 4, "big", signed=True)
        self.I.push(p1)
        self.assertEqual(self.I.stack[0], p1)
        self.assertEqual(self.I.sp, 1)
        with self.assertRaises(AssertionError):
            self.I.push((p1,), b'\x00\x00\x00\x02')
            self.I.sp = -1
            self.I.push(p1)
            self.I.sp = 10
            self.I.push(p1)
        self.I.sp = 0
        p2 = int.to_bytes(randint(-2**31, 2**31), 4, "big", signed=True)
        self.I.push(p1,p2)
        self.assertEqual(self.I.stack[0], p1)
        self.assertEqual(self.I.stack[1], p2)
        self.I.push((p1,p2)) # tuple
        self.assertEqual(self.I.stack[2], p1)
        self.assertEqual(self.I.stack[3], p2)
        self.I.stack, self.I.sp = [], 0
        with self.assertRaises(AssertionError): self.I.push(int32.to_bytes(0))

    def test_pop(self):
        self.I.stack = [None] * 10
        p1 = randint(0, 2**32)
        p2 = randint(0, 2**32)
        self.I.stack[0], self.I.stack[1] = p1, p2
        self.I.sp = 2
        self.assertEqual(self.I.pop(), p2)
        self.assertEqual(self.I.pop(), p1)
        with self.assertRaises(AssertionError):
            self.I.pop()
            self.I.sp = 11
            self.I.pop()
        self.I.sp = 3
        self.assertEqual(self.I.pop(), None)

        self.I.stack[0], self.I.stack[1] = p1, p2
        self.I.sp = 2
        self.assertEqual([p2, p1], self.I.pop(2))

        self.I.stack, self.I.sp = [], 0
        with self.assertRaises(AssertionError): self.I.pop()

    def test_SVTCA(self):
        self.I.gs = {}
        self.I.ops[0x00]() # Y
        self.assertEqual([self.I.gs["projection_vector"], self.I.gs["freedom_vector"]], [vec2(0, 1), vec2(0, 1)])
        self.I.ops[0x01]() # X
        self.assertEqual([self.I.gs["projection_vector"], self.I.gs["freedom_vector"]], [vec2(1, 0), vec2(1, 0)])
    
    def test_SPVTCA(self):
        self.I.gs = {}
        self.I.ops[0x02]() # Y
        self.assertEqual(self.I.gs["projection_vector"], vec2(0, 1))
        self.I.ops[0x03]() # x
        self.assertEqual(self.I.gs["projection_vector"], vec2(1, 0))
    
    def test_SFVTCA(self):
        self.I.gs = {}
        self.I.ops[0x04]() # Y
        self.assertEqual(self.I.gs["freedom_vector"], vec2(0, 1))
        self.I.ops[0x05]() # x
        self.assertEqual(self.I.gs["freedom_vector"], vec2(1, 0))
    
    # def test_SPVTL(self):
    #     """Set Projection Vector To Line"""
    #     self.I.gs = {}
    #     pointPairs = [
    #         [ 0,  0],
    #         [-1,  0],
    #         [ 0, -1],
    #         [-1, -1],
    #         [ 5,  2],
    #         [ 2,  3],
    #         [ 4,  1],
    #         [ 6,  1],
    #         [ 0,  7],
    #     ]
    #     zpPairs = [
    #         [0, 0]
    #         [0, 1]
    #         [1, 0]
    #         [1, 1]
    #     ]
    #     self.I.twilight = [
    #         vec2(0, 0),
    #         vec2(0, 1),
    #         vec2(5, 10),
    #         vec2(-50, 20),
    #         vec2(-200, -150),
    #         vec2(random(), random())
    #     ]
    #     self.I.glyf = None # TODO: setup glyf table
    #     for z in zpPairs: pass
            
    #     p1 = (random() * 2 - 1) * 10
    #     p2 = (random() * 2 - 1) * 10
    #     zp1 = randint(0,1)
    #     zp2 = randint(0,1)
    #     self.I.gs["zp1"], self.I.gs["zp2"] = zp1, zp2
    #     self.I.push(p1, p2)

    def test_SFVTL(self): pass
        # gs["freedom_vector"] = vec2(-(v0:=p1-p2).y, v0.x).normalize() if a else (p1-p2).normalize() # if a rotate counter clockwise 90 deg
    # def test_SPVFS(self): pass
    # def test_SFVFS(self): pass
    def test_GPV(self):
        pv = vec2(random(), random()).normalize()
        x, y = pv.components()
        self.I.gs = {"projection_vector": pv}
        self.I.stack, self.I.sp = [None]*2, 0
        self.I.ops[0x0C]()
        self.assertAlmostEqual(EF2Dot14(self.I.stack[0]), x, 3)
        self.assertAlmostEqual(EF2Dot14(self.I.stack[1]), y, 3)

    def test_GFV(self):
        fv = vec2(random(), random()).normalize()
        x, y = fv.components()
        self.I.gs = {"freedom_vector": fv}
        self.I.stack, self.I.sp = [None]*2, 0
        self.I.ops[0x0D]()
        self.assertAlmostEqual(EF2Dot14(self.I.stack[0]), x, 3)
        self.assertAlmostEqual(EF2Dot14(self.I.stack[1]), y, 3)


    def test_SFVTPV(self):
        v = vec2(random(), random()).normalize()
        self.I.gs = {"projection_vector": vec2(v.x, v.y)}
        self.I.ops[0x0E]()
        self.assertEqual(self.I.gs["projection_vector"], self.I.gs["freedom_vector"])
        self.I.gs["projection_vector"] += 2
        self.assertNotEqual(self.I.gs["projection_vector"], self.I.gs["freedom_vector"])

    def test_ISECT(self): pass
    # def test_SRP0(self): pass
    # def test_SRP1(self): pass
    # def test_SRP2(self): pass
    # def test_SZP0(self): pass
    # def test_SZP1(self): pass
    # def test_SZP2(self): pass
    def test_SZPS(self):
        inputs = [0, 1, -1, 10, 0.5]
        self.I.gs = {"zp0": None, "zp1": None, "zp2": None}
        for i in inputs:
            i = uint32(i)
            self.I.stack, self.I.sp = [uint32.to_bytes(i), None], 1
            if i in [0,1]:
                self.I.ops[self.I.opcodes["SZPS"]]()
                self.assertEqual(self.I.gs, {"zp0": i, "zp1": i, "zp2": i})
                self.assertEqual(self.I.sp, 0)
            else:
                with self.assertRaises(AssertionError): self.I.ops[self.I.opcodes["SZPS"]]()
    # def test_SLOOP(self): pass
    # def test_RTG(self): pass
    # def test_RTHG(self): pass
    # def test_SMD(self): pass
    def test_IF_ELSE_EIF(self):
        program = to_program(self.I, [
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
        ])

        treasure, toPop, trueV, falseV = int32.to_bytes(-1), int32.to_bytes(5), int32.to_bytes(9000), int32.to_bytes(0) 
        stacks = [
            [treasure, toPop, toPop, toPop, toPop,         trueV,  toPop, trueV ],
            [treasure, toPop, toPop, toPop, toPop, toPop,  falseV, toPop, trueV ],
            [treasure, toPop, toPop, toPop,        trueV,                 falseV],
            [treasure, toPop, toPop, toPop, toPop, falseV,                falseV]
        ]
        for s in stacks:
            self.I.stack = s
            self.I.sp = len(s)
            self.I.run_program(program, "fpgm")
            self.assertEqual(self.I.stack[self.I.sp - 1], treasure)

        # unterminated if statement
        program = to_program(self.I, [
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
        ])

        # unterminated if statement inside function definition
        program = to_program(self.I, [
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
        ])
        for s in stacks:
            self.I.maxp = maxp
            self.I.functions = [None]
            self.I.stack = s + [uint32.to_bytes(1), uint32.to_bytes(1)]
            self.I.sp = len(s)
            with self.assertRaises(AssertionError): self.I.run_program(program, "fpgm")

        # if IF and ELSE followed by values that are the same as relevant opcodes
        # test is somewhat reduced compared to the one in test_FDEF_ENDF_CALL_LOOPCALL because it builds on the same stuff
        # IF
        program  = to_program(self.I, [
            "IF",
                "PUSHB(0,)",
                "IF",
            "EIF"
        ])
        self.I.stack, self.I.sp = [uint32.to_bytes(1)], 1
        self.I.run_program(program, "fpgm")
        self.assertEqual(self.I.stack, [uint32.to_bytes(self.I.opcodes["IF"])])
        # ELSE
        program  = to_program(self.I, [
            "IF",
            "ELSE",
                "PUSHB(1,)",
                "ELSE",
                "EIF",
            "EIF"
        ])
        self.I.stack, self.I.sp = uint32_stack([0, None ]), 1
        self.I.run_program(program, "fpgm")
        self.assertEqual(self.I.stack, [uint32.to_bytes(self.I.opcodes["ELSE"]), uint32.to_bytes(self.I.opcodes["EIF"])])

    # def test_JMPR(self): pass
    # def test_SCVTCI(self): pass
    # def test_SSWCI(self): pass
    # def test_SSW(self): pass

    def test_DUP(self):
        self.I.stack = [b'\x00\x00\x00\x00', None, None]
        self.I.sp = 1
        self.I.ops[self.I.opcodes["DUP"]]()
        self.assertEqual(self.I.stack, [b'\x00\x00\x00\x00', b'\x00\x00\x00\x00', None])
        self.I.stack, self.I.sp = [int32.to_bytes(0)], 1
        with self.assertRaises(AssertionError): self.I.ops[self.I.opcodes["DUP"]]()
        self.I.stack, self.I.sp = [int32.to_bytes(0), None], 1
        self.I.ops[self.I.opcodes["DUP"]]()
        self.assertEqual(self.I.stack, int32_stack([0, 0]))
        self.assertEqual(self.I.sp, 2)
        self.I.stack, self.I.sp = [], 0
        with self.assertRaises(AssertionError): self.I.ops[self.I.opcodes["DUP"]]()

    # def test_CLEAR(self): pass

    def test_SWAP(self):
        p1, p2 = b'\x00\x00\x00\x05', b'\x00\x00\x00\x07'
        self.I.stack = [None, p1, p2]
        self.I.sp = 3
        self.I.ops[0x23]()
        self.assertEqual(self.I.stack, [None, p2, p1])

    def test_DEPTH(self):
        self.I.stack = [b'\x00\x00\x00\x02', b'\x00\x00\x00\x02', None, None]
        self.I.sp = 2
        self.I.ops[0x24]()
        self.assertEqual(self.I.stack, [b'\x00\x00\x00\x02', b'\x00\x00\x00\x02', b'\x00\x00\x00\x02', None])

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
            self.I.ops[self.I.opcodes["CINDEX"]]()
            self.assertEqual(self.I.stack, int32_stack(c[:4] + [c[-c[-1] - 1]]))
            self.assertEqual(self.I.sp, len(c))
        for c in invalid_cases: 
            self.I.stack, self.I.sp = int32_stack(c), len(c)
            with self.assertRaises(AssertionError): self.I.ops[self.I.opcodes["CINDEX"]]()


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
            self.I.ops[self.I.opcodes["MINDEX"]]()
            self.assertEqual(self.I.stack, int32_stack(r))
            self.assertEqual(self.I.sp, len(c) - 1)
        for c in invalid_cases: 
            self.I.stack, self.I.sp = int32_stack(c), len(c)
            with self.assertRaises(AssertionError): self.I.ops[self.I.opcodes["MINDEX"]]()

    def test_ALIGNPTS(self): pass
    def test_UTP(self): pass

    def test_FDEF_ENDF_CALL_LOOPCALL(self):
        self.I.maxp = maxp()
        program = to_program(self.I, [
            "FDEF",
                "POP",
            "ENDF",
            "FDEF",
                "POP",
                "POP",
            "ENDF",
            "CALL",
            "LOOPCALL"
        ])
        self.I.functions = [None, None]
        v3, v0, v4 = int32.to_bytes(3), int32.to_bytes(0), int32.to_bytes(4)
        v1, v5, v_5 = int32.to_bytes(1), int32.to_bytes(5), int32.to_bytes(-5)
        self.I.stack, self.I.sp = [v3, v3, v3, v3, v4, v0, v3, v0, v1, v0], 10
        self.I.run_program(program, "fpgm")
        self.assertEqual(self.I.functions, [{"context": "fpgm", "ip": 1}, {"context": "fpgm", "ip": 4}])
        self.assertEqual(self.I.sp, 0)

        self.I.stack, self.I.sp = [v3, v3, v3, v3, v0, v0, v3, v0, v1, v0], 10
        self.I.run_program(program, "fpgm")
        self.assertEqual(self.I.sp, 4)

        self.I.stack, self.I.sp = [v3, v3, v3, v3, v4, v0, v3, v0, v1], 9
        with self.assertRaises(AssertionError): self.I.run_program(program, "fpgm")

        self.I.stack, self.I.sp = [v3, v3, v3, v3, v5, v0, v3, v0, v0], 9
        with self.assertRaises(AssertionError): self.I.run_program(program, "fpgm")

        self.I.stack, self.I.sp = [v3, v3, v3, v3, v5, v0, v3, v1, v0], 9
        with self.assertRaises(AssertionError): self.I.run_program(program, "fpgm")

        self.I.stack, self.I.sp = [v3, v3, v3, v3, v_5, v0, v3, v0, v0], 9
        with self.assertRaises(AssertionError): self.I.run_program(program, "fpgm")

        program2 = [ self.I.opcodes["CALL"], self.I.opcodes["POP"] ]
        self.I.stack, self.I.sp = [v3, v3, v0, v3, v1, v0, v3, v0, v1, v0], 10
        self.I.run_program(program, "fpgm")
        self.assertEqual(self.I.sp, 3)
        self.I.run_program(program2, "prep")
        self.assertEqual(self.I.sp, 0)

        self.I.stack, self.I.sp = [v3, v3, v0, v3, v1, v1, v3, v0, v1, v0], 10
        self.I.run_program(program, "fpgm")
        self.assertEqual(self.I.sp, 2)

        # test differentiation between op and value in FDEF
        for n in range(8):
            # test with PUSHB
            n_true = n+1
            program = to_program(self.I, [
                "FDEF",
                    f"PUSHB({n},)"] +
                    ["FDEF"] * n_true + # should be read as values for push
                ["ENDF"]
            )
            self.I.stack, self.I.sp = uint32_stack([0]), 1
            self.I.run_program(program, "fpgm")
            # test with PUSHW
            program = to_program(self.I, [
                "FDEF",
                    f"PUSHW({n},)"] +
                    ["FDEF"] * n_true * 2 + # should be read as values for push
                ["ENDF"]
            )
            self.I.stack, self.I.sp = uint32_stack([0]), 1
            self.I.run_program(program, "fpgm")
        for i in range(4,20):
            # test with NPUSHB
            program = to_program(self.I, [
                "FDEF",
                    "NPUSHB",
                    uint8.to_bytes(i)] +
                    ["FDEF"] * i + # should be read as values for push
                ["ENDF"]
            )
            self.I.stack, self.I.sp = uint32_stack([0]), 1
            self.I.run_program(program, "fpgm")
            # test with NPUSHW
            program = to_program(self.I, [
                "FDEF",
                    "NPUSHW",
                    uint8.to_bytes(i)] +
                    ["FDEF"] * i * 2 + # should be read as values for push
                ["ENDF"]
            )
            self.I.stack, self.I.sp = uint32_stack([0]), 1
            self.I.run_program(program, "fpgm")

    def test_MDAP(self): pass
    def test_IUP(self): pass
    def test_SHP(self): pass
    def test_SHC(self): pass
    def test_SHZ(self): pass
    def test_SHPIX(self): pass
    def test_IP(self): pass
    def test_MSIRP(self): pass
    def test_ALIGNRP(self): pass
    # def test_RTDG(self): pass
    def test_MIAP(self): pass

    def test_NPUSHB(self):
        self.I.stack, self.I.sp = [None, None, None], 0
        program = to_program(self.I, ["NPUSHB", 0x2, 0x28, 0xf2])
        self.I.run_program(program, "fpgm")
        self.assertEqual(self.I.stack, [b'\x00\x00\x00\x28', b'\x00\x00\x00\xf2', None])
        self.assertEqual(self.I.sp, 2)

        self.I.stack, self.I.sp = [None, None, None], 0
        program = to_program(self.I, ["NPUSHB", 0])
        self.I.run_program(program, "fpgm")
        self.assertEqual(self.I.stack, [None, None, None])
        self.assertEqual(self.I.sp, 0)

        self.I.stack, self.I.sp = [None, None, None], 0
        program = to_program(self.I, ["NPUSHB", 0x3, 0x28, 0xf2])
        with self.assertRaises(AssertionError): self.I.run_program(program, "fpgm")

    def test_NPUSHW(self):
        self.I.stack, self.I.sp = [None, None, None], 0
        program = to_program(self.I, ["NPUSHW", 0x2, 0x28, 0xf2, 0x5d, 0xa])
        self.I.run_program(program, "fpgm")
        self.assertEqual(self.I.stack, [b'\x00\x00\x28\xf2', b'\x00\x00\x5d\x0a', None])
        self.assertEqual(self.I.sp, 2)

        self.I.stack, self.I.sp = [None, None, None], 0
        program = to_program(self.I, ["NPUSHW", 0x0])
        self.I.run_program(program, "fpgm")
        self.assertEqual(self.I.stack, [None, None, None])
        self.assertEqual(self.I.sp, 0)

        self.I.stack, self.I.sp = [None, None, None], 0
        program = to_program(self.I, ["NPUSHW", 0x2, 0x28, 0xf2, 0x5d])
        with self.assertRaises(AssertionError): self.I.run_program(program, "fpgm")

    def test_WS_RS(self):
        # WS
        self.I.stack, self.I.sp = uint32_stack([6, 0, 0, 5]), 4
        self.I.maxp, self.I.storage = maxp(), [None]
        self.I.ops[self.I.opcodes["WS"]]()
        self.assertEqual(self.I.storage[0], uint32(5))
        self.assertEqual(self.I.sp, 2)
        with self.assertRaises(AssertionError): self.I.ops[self.I.opcodes["WS"]]()

        # RS
        self.I.stack, self.I.sp = uint32_stack([6, 0, 0, 5]), 4
        self.I.maxp, self.I.storage = maxp(), [None]
        self.I.ops[self.I.opcodes["WS"]]()
        self.assertEqual(self.I.storage[0], uint32(5))
        self.assertEqual(self.I.sp, 2)
        self.I.ops[self.I.opcodes["RS"]]()
        self.assertEqual(self.I.stack[:2], [uint32.to_bytes(6), uint32.to_bytes(self.I.storage[0])])
        self.assertEqual(self.I.sp, 2)

        self.I.storage = [None, uint32(90)]
        self.I.stack, self.I.sp = [uint32.to_bytes(4), uint32.to_bytes(1), None], 2
        self.I.ops[self.I.opcodes["RS"]]()
        self.assertEqual(self.I.stack, [uint32.to_bytes(4), uint32.to_bytes(90), None])
        self.assertEqual(self.I.sp, 2)
        with self.assertRaises(AssertionError): self.I.ops[self.I.opcodes["RS"]]()

    def test_WCVTP(self):
        self.I.stack, self.I.sp = [uint32.to_bytes(5), F26Dot6.to_bytes(-1526.5), uint32.to_bytes(0), F26Dot6.to_bytes(15.125)], 4
        self.I.cvt = [None] * 2
        self.I.ops[self.I.opcodes["WCVTP"]]()
        self.assertEqual(self.I.cvt[0], F26Dot6(15.125))
        self.assertEqual(self.I.sp, 2)

        # RCVT
        self.I.push(uint32.to_bytes(0))
        self.I.ops[self.I.opcodes["RCVT"]]()
        self.assertEqual(self.I.stack[3], F26Dot6.to_bytes(15.125))
        self.assertEqual(self.I.sp, 3)

        with self.assertRaises(AssertionError): self.I.ops[self.I.opcodes["WCVTP"]]()
        
    def test_RCVT(self):
        self.I.cvt = [F26Dot6(i) for i in self.inputs]
        for i in range(len(self.I.cvt)):
            self.I.stack, self.I.sp = [uint32.to_bytes(i), None], 1
            self.I.ops[self.I.opcodes["RCVT"]]()
            self.assertEqual(self.I.stack, [F26Dot6.to_bytes(self.I.cvt[i]), None])
            self.assertEqual(self.I.sp, 1)
        for i in self.inputs:
            v = int32(i)
            self.I.stack, self.I.sp = [int32.to_bytes(v), None], 1
            if len(self.I.cvt) > v >= 0:
                self.I.ops[self.I.opcodes["RCVT"]]()
                self.assertEqual(self.I.stack, [F26Dot6.to_bytes(self.I.cvt[v]), None])
                self.assertEqual(self.I.sp, 1)
            else:
                with self.assertRaises(AssertionError): self.I.ops[self.I.opcodes["RCVT"]]()

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
                self.I.ops[self.I.opcodes[f"GC({a},)"]]()
                self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(res))
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
            self.I.ops[self.I.opcodes[f"MD({a},)"]]()
            res = self.I.orthogonal_projection(p2, self.I.gs["projection_vector"]) - self.I.orthogonal_projection(p1, self.I.gs["projection_vector"])
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(self.I.FU_to_px(res)))
        
        for a in [0,1]:
            self.I.gs = {"zp0": 1, "zp1": 1, "projection_vector": vec2(1, 0)} # test glyph zone
            for i in range(0, len(self.I.g.x)-1, 2): test()
            self.I.gs["zp0"] = self.I.gs["zp1"] = 0 # test twilight zone
            for i in range(0, len(self.I.twilight)-1, 2): test()

    def test_MPPEM(self):
        self.I.fontsize, self.I.dpi = 12, 144
        self.I.stack, self.I.sp = [None], 0
        self.I.ops[self.I.opcodes["MPPEM"]]()
        self.assertEqual(uint32(self.I.stack[0]), int(12 * 144 / 72))
    def test_MPS(self): pass
    # def test_FLIPON(self): pass
    # def test_FLIPOFF(self): pass
    def test_DEBUG(self): pass
    def test_LT(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(uint32.to_bytes, (i1, i2))), 2
            self.I.ops[self.I.opcodes["LT"]]()
            self.assertEqual(self.I.stack[0], uint32.to_bytes(int(uint32(i1) < uint32(i2))))
    def test_LTEQ(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(uint32.to_bytes, (i1, i2))), 2
            self.I.ops[self.I.opcodes["LTEQ"]]()
            self.assertEqual(self.I.stack[0], uint32.to_bytes(int(uint32(i1) <= uint32(i2))))
    def test_GT(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(uint32.to_bytes, (i1, i2))), 2
            self.I.ops[self.I.opcodes["GT"]]()
            self.assertEqual(self.I.stack[0], uint32.to_bytes(int(uint32(i1) > uint32(i2))))
    def test_GTEQ(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(uint32.to_bytes, (i1, i2))), 2
            self.I.ops[self.I.opcodes["GTEQ"]]()
            self.assertEqual(self.I.stack[0], uint32.to_bytes(int(uint32(i1) >= uint32(i2))))
    def test_EQ(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(uint32.to_bytes, (i1, i2))), 2
            self.I.ops[self.I.opcodes["EQ"]]()
            self.assertEqual(self.I.stack[0], uint32.to_bytes(int(uint32(i1) == uint32(i2))))
    def test_NEQ(self): 
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(uint32.to_bytes, (i1, i2))), 2
            self.I.ops[self.I.opcodes["NEQ"]]()
            self.assertEqual(self.I.stack[0], uint32.to_bytes(int(uint32(i1) != uint32(i2))))
    def test_ODD(self): pass
    def test_EVEN(self): pass
    # def test_IF(self): pass # tested in test_IF_ELSE_EIF already
    # def test_EIF(self): pass # tested in test_IF_ELSE_EIF already
    def test_AND(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(uint32.to_bytes, (i1, i2))), 2
            self.I.ops[self.I.opcodes["AND"]]()
            self.assertEqual(self.I.stack[0], uint32.to_bytes(int(uint32(i1) != 0 and uint32(i2) != 0)))
            self.assertEqual(self.I.sp, 1)

    def test_OR(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack, self.I.sp = list(map(uint32.to_bytes, (i1, i2))), 2
            self.I.ops[self.I.opcodes["OR"]]()
            self.assertEqual(self.I.stack[0], uint32.to_bytes(int(uint32(i1) != 0 or uint32(i2) != 0)))
            self.assertEqual(self.I.sp, 1)

    def test_NOT(self):
        for i in self.inputs:
            self.I.stack, self.I.sp = [uint32.to_bytes(i)], 1
            self.I.ops[self.I.opcodes["NOT"]]()
            self.assertEqual(self.I.stack[0], uint32.to_bytes(int(not uint32(i))))
            self.assertEqual(self.I.sp, 1)

    def test_DELTAP1(self): pass
    def test_SDB(self):
        inputs = [2**32-1, 0, 1, -1231, 123.45]
        for i in inputs:
            self.I.gs = {"delta_base": 9}
            self.I.stack, self.I.sp = [uint32(i), None], 1
            self.I.ops[self.I.opcodes["SDB"]]()
            self.assertEqual(self.I.gs["delta_base"], uint32(i))
    def test_SDS(self):
        inputs = [2**32-1, 0, 1, -1231, 123.45]
        for i in inputs:
            self.I.gs = {"delta_shift": 3}
            self.I.stack, self.I.sp = [uint32(i), None], 1
            self.I.ops[self.I.opcodes["SDS"]]()
            self.assertEqual(self.I.gs["delta_shift"], uint32(i))

    # NOTE: F26Dot6's low precision easily causes errors in arithemtic even at two decimal points
    def test_ADD(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack = list(map(lambda x: F26Dot6.to_bytes(x), (i1, i2)))
            self.I.sp = 2
            self.I.ops[self.I.opcodes["ADD"]]()
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(sum((F26Dot6(i1), F26Dot6(i2)))))
            self.assertEqual(self.I.sp, 1)

    def test_SUB(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack = list(map(lambda x: F26Dot6.to_bytes(x), (i1, i2)))
            self.I.sp = 2
            self.I.ops[self.I.opcodes["SUB"]]()
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(F26Dot6(i1) - F26Dot6(i2)))
            self.assertEqual(self.I.sp, 1)

    def test_DIV(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack = list(map(lambda x: F26Dot6.to_bytes(x), (i1, i2)))
            self.I.sp = 2
            self.I.ops[self.I.opcodes["DIV"]]()
            if F26Dot6(i2) == 0:
                self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6((-1 if i1 < 0 else 1) * 0xFFFFFFFF))
            else:
                self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(F26Dot6(i1) / F26Dot6(i2)))
        self.I.stack, self.I.sp = [F26Dot6.to_bytes(1), F26Dot6.to_bytes(0)], 2
        self.I.ops[self.I.opcodes["DIV"]]()
        self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(0xFFFFFFFF))

    def test_MUL(self):
        for i1, i2 in zip(self.inputs, self.inputs[1:]):
            self.I.stack = list(map(lambda x: F26Dot6.to_bytes(x), (i1, i2)))
            self.I.sp = 2
            self.I.ops[self.I.opcodes["MUL"]]()
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(F26Dot6(i1) * F26Dot6(i2)))

    def test_ABS(self):
        for i in self.inputs:
            self.I.stack = [F26Dot6.to_bytes(i)]
            self.I.sp = 1
            self.I.ops[self.I.opcodes["ABS"]]()
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(abs(i)))

    def test_NEG(self):
        for i in self.inputs:
            self.I.stack = [F26Dot6.to_bytes(i)]
            self.I.sp = 1
            self.I.ops[self.I.opcodes["NEG"]]()
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(-F26Dot6(i)))

    def test_FLOOR(self):
        for i in self.inputs:
            self.I.stack = [F26Dot6.to_bytes(i)]
            self.I.sp = 1
            self.I.ops[self.I.opcodes["FLOOR"]]()
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(math.floor(F26Dot6(i))))

    def test_CEILING(self):
        for i in self.inputs:
            self.I.stack = [F26Dot6.to_bytes(i)]
            self.I.sp = 1
            self.I.ops[self.I.opcodes["CEILING"]]()
            self.assertEqual(F26Dot6(self.I.stack[0]), F26Dot6(math.ceil(F26Dot6(i))))

    def test_ROUND(self):
        rounding = ["RTHG", "RTG", "RTDG", "RDTG", "RUTG", "ROFF"] #, "SROUND", "S45ROUND"]
        self.I.gs = {"round_state": False, "minimum_distance": 0}
        for i in self.inputs:
            i = F26Dot6(i)
            for r in rounding:
                self.I.ops[self.I.opcodes[r]]()
                self.I.stack, self.I.sp = [F26Dot6.to_bytes(i)], 1
                self.I.ops[self.I.opcodes["ROUND(0,)"]]()
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
            self.I.stack, self.I.sp = uint32_stack([5, 1526, 0]) + [uint32.to_bytes(i)], 4
            self.I.fontsize = 12
            self.I.dpi = 144
            self.I.head = head
            self.I.cvt = [None] * 2
            self.I.ops[self.I.opcodes["WCVTF"]]()
            self.assertEqual(self.I.cvt[0], F26Dot6(uint32(i) * ((self.I.fontsize * self.I.dpi) / (72*self.I.head.unitsPerEM))))
            self.assertEqual(self.I.sp, 2)

            with self.assertRaises(AssertionError): self.I.ops[self.I.opcodes["WCVTP"]]()

    def test_DELTAP2(self): pass
    def test_DELTAP3(self): pass
    def test_DELTAC1(self): pass
    def test_DELTAC2(self): pass
    def test_DELTAC3(self): pass
    # def test_SROUND(self): pass
    # def test_S45ROUND(self): pass
    def test_JROT(self): pass
    def test_JROF(self): pass
    # def test_ROFF(self): pass
    # def test_RUTG(self): pass
    # def test_RDTG(self): pass
    def test_SANGW(self): pass
    # def test_AA(self): pass
    def test_FLIPPT(self): pass
    def test_FLIPRGON(self): pass
    def test_FLIPRGOFF(self): pass
    def test_SCANCTRL(self): pass
    def test_SDPVTL(self): pass
    def test_GETINFO(self):
        selector = int32.to_bytes(-1) # all 1s in binary
        self.I.stack, self.I.sp = [selector], 1
        self.I.ops[self.I.opcodes["GETINFO"]]()
        self.assertEqual(self.I.stack, [int32.to_bytes(2)]) # only engine version is returned as 2, rest is 0
        self.assertEqual(self.I.sp, 1)
        self.I.stack, self.I.sp = [uint32.to_bytes(0)], 1 # get nothing
        self.I.ops[self.I.opcodes["GETINFO"]]()
        self.assertEqual(self.I.stack, [uint32.to_bytes(0)])
        self.I.stack, self.I.sp = [uint32.to_bytes(1)], 1 # get only engine version
        self.I.ops[self.I.opcodes["GETINFO"]]()
        self.assertEqual(self.I.stack, [uint32.to_bytes(2)])

    def test_IDEF(self): pass

    def test_ROLL(self):
        valid_cases = [
            [2, None, 0, 0xff, 0xd3, None],
            [2, 0, 0xff, 0xd3, None],
            [2, 0, 0xff, None]
        ]
        results = [
            [2, None, 0xff, 0xd3, 0, 3],
            [2, 0xff, 0xd3, 0, 3],
            [0, 0xff, 2, 3]
        ]
        invalid_case = [2,0]
        for c,r in zip(valid_cases, results):
            self.I.stack, self.I.sp = int32_stack(c), len(c) - 1
            self.I.ops[self.I.opcodes["ROLL"]]()
            self.assertEqual(self.I.stack, int32_stack(r))
            self.assertEqual(self.I.sp, len(c) - 1)
        self.I.stack, self.I.sp = int32_stack(invalid_case), len(invalid_case)
        with self.assertRaises(AssertionError): self.I.ops[self.I.opcodes["ROLL"]]()

    def test_MAX(self): pass
    def test_MIN(self): pass
    def test_SCANTYPE(self): pass
    def test_INSTCTRL(self): pass
    def test_PUSHB(self):
        self.I.stack, self.I.sp = [None, None, None], 0
        program = to_program(self.I, ["PUSHB(0,)", 0x2])
        self.I.run_program(program, "fpgm")
        self.assertEqual(self.I.stack, [b'\x00\x00\x00\x02', None, None])
        self.assertEqual(self.I.sp, 1)

        self.I.stack, self.I.sp = [None, None, None], 0
        program = to_program(self.I, ["PUSHB(2,)", 0, 0xff, 0xd3])
        self.I.run_program(program, "fpgm")
        self.assertEqual(self.I.stack, [b'\x00\x00\x00\x00', b'\x00\x00\x00\xff', b'\x00\x00\x00\xd3'])
        self.assertEqual(self.I.sp, 3)

        self.I.stack, self.I.sp = [None, None, None], 0
        program = to_program(self.I, ["PUSHB(3,)", 0x3, 0x28, 0xf2])
        with self.assertRaises(AssertionError): self.I.run_program(program, "fpgm")

    def test_PUSHW(self):
        self.I.stack, self.I.sp = [None, None, None], 0
        program = to_program(self.I, ["PUSHW(0,)", 0x02, 0x42])
        self.I.run_program(program, "fpgm")
        self.assertEqual(self.I.stack, [b'\x00\x00\x02\x42', None, None])
        self.assertEqual(self.I.sp, 1)

        self.I.stack, self.I.sp = [None, None, None], 0
        program = to_program(self.I, ["PUSHW(1,)", 0x00, 0xff, 0xd3, 0x23])
        self.I.run_program(program, "fpgm")
        self.assertEqual(self.I.stack, [b'\x00\x00\x00\xff', b'\x00\x00\xd3\x23', None])
        self.assertEqual(self.I.sp, 2)

        self.I.stack, self.I.sp = [None, None, None], 0
        program = to_program(self.I, ["PUSHW(2,)", 0x03, 0x28, 0xf2, 0xff, 0xb0])
        with self.assertRaises(AssertionError): self.I.run_program(program, "fpgm")

    def test_MDRP(self): pass
    def test_MIRP(self): pass


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
        for f in fonts: Interpreter(fontfile=f, test=True)

if __name__ == "__main__": unittest.main()