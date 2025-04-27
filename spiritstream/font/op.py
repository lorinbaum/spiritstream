from typing import Union, List, Callable
from spiritstream.vec import vec2
from spiritstream.dtype import *
import math
from dataclasses import dataclass
from functools import partial
import spiritstream.font.ttf as ttf

"""GEOMETRY OPS"""

def FU_to_px(I:"ttf.Interpreter", v:Union[float, int]) -> float: return v * (I.fontsize * I.dpi) / (72 * I.head.unitsPerEM)
def move(I:"ttf.Interpreter", point_index:int, zone_pointer:str, coordinate:Union[int, float]=None, x=None, y=None) -> None:
    """If coordinate is set, moves point p along the freedom vector until it has the value of coordinate when projected on the projection_vector. If coordinate is not set, moves p to x,y."""
    assert zone_pointer in ["zp0", "zp1", "zp2"]
    zp = I.gs[zone_pointer]
    if coordinate != None and x == y == None:
        assert I.gs["projection_vector"].dot(I.gs["freedom_vector"]) != 0
        p = I.get_point(point_index, zone_pointer, "fitted")
        p = vec2(p.x, p.y)
        current_coordinate = orthogonal_projection(p, I.gs["projection_vector"])
        pos = p + (coordinate-current_coordinate) / orthogonal_projection(I.gs["freedom_vector"], I.gs["projection_vector"]) * I.gs["freedom_vector"]
    elif coordinate == None and x != None and y != None: pos = vec2(x, y)
    else: raise AssertionError(f"Either coordinate or x and y must be defined, got {coordinate=}, {x=}, {y=}")
    if zp == 0: I.twilight[point_index] = pos
    else:
        I.g.touched[point_index] = True
        I.g.fitted_x[point_index], I.g.fitted_y[point_index] = pos.components()
def orthogonal_projection(v1:vec2, v2:vec2) -> float:
    """projects v1 orthogonally onto v2. v2 is treated as a "ruler", so after projection v1 is a scalar along v2. this scalar can be negative, unlike magnitude. To get vector from this, simply multiply by vec2."""
    return (v1.dot(v2) / (v2.dot(v2)))
def ALIGNPTS(I:"ttf.Interpreter"): raise NotImplementedError # TODO: access glyph points
def MDAP(I:"ttf.Interpreter", a):
    """Move Direct Absolute Point"""
    pi = uint32(pop(I))
    if a:
        p = I.get_point(pi, "zp0", "fitted")
        coordinate = orthogonal_projection(vec2(p.x, p.y), I.gs["projection_vector"])
        move(I, pi, "zp0", _round(I, coordinate))
    I.gs["rp0"] = I.gs["rp1"] = pi
    if I.gs["zp0"] == 1: I.g.touched[pi] = True
def UTP(I:"ttf.Interpreter"): raise NotImplementedError
def IUP(I:"ttf.Interpreter", a):
    """
    Interpolate Untouched Points through the outline
    For each contour in the glyph, find touched-untouched-touched point sequences.
    If a is 0, consider y direction, if a is 1, consider x direction.
    If the untouched point was between the touched ones on the relevant axis, move it on that axis such that their relationship from the original outline is restored.
    Else determine the movement of the closer touched point and move the untouched point the same way, though only limited to the relevant axis.
    IUP does not "touch" the points it moves.
    """
    # raise NotImplementedError
    assert I.gs["zp2"] == 1, "zp2 must be set to 1 before IUP instruction. IUP in twilight zone unsupported"
    start = 0
    for ep in I.g.endPtsContours:
        pts = [I.get_point(i, "zp2", "original") for i in [ep] + list(range(start, ep + 1)) + [start]] # add last and first points to allow full detection of a 3 element pattern
        for i, (p1, p2, p3) in enumerate(zip(pts, pts[1:], pts[2:])):
            if p1.touched and not p2.touched and p3.touched:
                point_index = i + start # middle point (p2)
                p2x, p2y = p2.x, p2.y # storing original data
                p1, p2, p3 = (p1.x, p2.x, p3.x) if a else (p1.y, p2.y, p3.y) # get relevant coordinate data only
                mi = min(p1, p2, p3)
                mx = max(p1, p2, p3)
                if mi == mx: continue
                if mx > p2 > mi: # p2 between the two points
                    p1, p2, p3 = [(p - mi) / (mx - mi) for p in [p1, p2, p3]] # normalize
                    # get points after fitting and move p2 to maintain ratio
                    p1_index = point_index - 1 if point_index > start else ep
                    p3_index = point_index + 1 if point_index < ep else start
                    p1n, p3n = I.get_point(p1_index, "zp2", "fitted"), I.get_point(p3_index, "zp2", "fitted") 
                    p1n, p3n = (p1n.x, p3n.x) if a else (p1n.y, p3n.y) # get relevant coordinate data only
                    new_c = (p3n + (p1n - p3n) * p2) if p1 > p3 else (p1n + (p3n - p1n) * p2)
                    x, y = (new_c, p2y) if a else (p2x, new_c)
                else:
                    closest_point_index = i-1 if abs(p1-p2) < abs(p3-p2) else i+1
                    closest_point_index %= len(pts) - 2 # -2 for the two duplicated points
                    closest_point_index += start
                    # get difference before and after fitting
                    before = I.get_point(closest_point_index, "zp2", "original")
                    after = I.get_point(closest_point_index, "zp2", "fitted")
                    x, y = (p2x + (after.x - before.x), p2y) if a else (p2x, p2y + (after.y - before.y))
                move(I, point_index, "zp2", x=x, y=y)
                I.g.touched[point_index] = False # untouch after move. IUP does not touch
        start = ep + 1
def SHP(I:"ttf.Interpreter", a): raise NotImplementedError
def SHC(I:"ttf.Interpreter", a):
    """
    Shift contour by the last point
    Shifts contour c in zone zp2 (I'm assuming the contour that point c is on) by how how much the reference point has been shifted. if a == 0 that's rp2 in zp1, if a == 1 it's rp1 in zp0
    """
    assert I.gs["zp2"] == 1, "Unsure what to do in SHC instruction in twilight zone"
    assert a in [0,1]
    rp = I.gs["rp2"] if a == 0 else I.gs["rp1"]
    zone = "zp1" if a == 0 else "zp0"
    original_point = I.get_point(rp, zone, "original")
    moved_point = I.get_point(rp, zone, "fitted")
    d1 = orthogonal_projection(vec2(original_point.x, original_point.y), I.gs["projection_vector"])
    d2 = orthogonal_projection(vec2(moved_point.x, moved_point.y), I.gs["projection_vector"])
    d = d2 - d1
    c = uint32(pop(I))
    assert c < I.g.endPtsContours[-1] + 1
    start = 0
    for ep in I.g.endPtsContours:
        if start <= c <= ep:
            contour = list(range(start, ep+1))
            break
        start = ep + 1
    assert contour != None
    for pi in contour:
        if I.gs[zone] == 1 and rp == pi: continue # skip reference point if in contour c
        p = I.get_point(pi, "zp2", "fitted")
        coordinate = orthogonal_projection(vec2(p.x, p.y), I.gs["projection_vector"])
        # print(f"SHC{a} moving {pi} as part of contour ({len(contour)=}) in {I.gs['zp2']} by {d} to coordinate {coordinate + d} given {I.gs['projection_vector']} and point . Point stayed the same = {(p.x, p.y) == ((pnew:=I.get_point(pi, zone, 'fitted')).x, pnew.y)}")
        # print("|   original point:", p.x, p.y, "confirmation", (npoint:=I.get_point(pi, "zp2", "fitted")).x, npoint.y)
        move(I, pi, "zp2", coordinate + d)
    
def SHZ(I:"ttf.Interpreter", a): raise NotImplementedError
def SHPIX(I:"ttf.Interpreter"):
    """
    SHift point by a PIXel amount
    Pop distance to move from stack. For as many times as specified in loop variable in graphics state, pop a point index from the stack and move that point by distance along the freedom vector.
    """
    d = F26Dot6(pop(I))
    assert I.gs["loop"] > 0
    for _ in range(I.gs["loop"]):
        pi = uint32(pop(I))
        np = I.get_point(pi, "zp2", "fitted")
        np = vec2(np.x, np.y) + I.gs["freedom_vector"] * d
        move(I, pi, "zp2", x=np.x, y=np.y)
    I.gs["loop"] = 1
def IP(I:"ttf.Interpreter"): raise NotImplementedError
def MSIRP(I:"ttf.Interpreter", a): raise NotImplementedError
def ALIGNRP(I:"ttf.Interpreter"):
    """
    Align Relative Point
    Reduces the distance between rp0 and point p to zero. Pops as many points as loop from graphics state demands. Since distance is measured along the projection_vector and movement is along the freedom_vector, the effect of the instruction is to align points.
    """
    rp0 = I.get_point(I.gs["rp0"], "zp0", "fitted")
    rp0_coord = orthogonal_projection(vec2(rp0.x, rp0.y), I.gs["projection_vector"])
    assert I.gs["loop"] > 0
    for _ in range(I.gs["loop"]):
        pi = uint32(pop(I))
        # print(f"ALIGNRP moving {pi} in {I.gs['zp1']} to coordinate of point {I.gs['rp0']}: {rp0_coord}")
        move(I, pi, "zp1", rp0_coord)
    I.gs["loop"] = 1
def MIAP(I:"ttf.Interpreter", a):
    """
    Move Indirect Absolute Point
    Pop n, then p from stack. Move point p along the freedom vector until, when projected onto the projection vector, it will have a value equal to cvt[n]. If a is 1, additionally consider control_value_cut_in and round the value before setting the point. For this, the coordinate of point p from the original outline, as projected on the projection vector is compared to cvt[n]. If the absolute difference between them is less than control value cut in, original coordinate is used to move the point p, else cvt[n]
    """
    assert I.gs["projection_vector"].dot(I.gs["freedom_vector"]) != 0
    c, pi = I.cvt[uint32(pop(I))], uint32(pop(I))
    if a:
        raise NotImplementedError
        p = I.get_point(pi, "zp0", "original")
        coordinate = FU_to_px(I, orthogonal_projection(vec2(p.x, p.y), I.gs["projection_vector"]))
        target = c if abs(coordinate - c) <= I.gs["control_value_cut_in"] else coordinate
        v = _round(I, target)
        move(I, pi, "zp0", v)
    else: move(I, pi, "zp0", c)
    I.gs["rp0"] = I.gs["rp1"] = pi
def GC(I:"ttf.Interpreter", a):
    """Get Coordinate projected onto the projection vector"""
    assert a in [0,1]
    p = I.get_point(uint32(pop(I)), "zp2", "original" if a else "fitted")
    v = orthogonal_projection(vec2(p.x, p.y), I.gs["projection_vector"])
    push(I, F26Dot6.to_bytes(v))
def SCFS(I:"ttf.Interpreter"): raise NotImplementedError
def MD(I:"ttf.Interpreter", a):
    """
    Measure distance
    Pop two uint32 (point indices p1 and p2) from the stack. p1 uses zp1, p2 uses zp0. Project the points onto the projection vector. Then measure distance from point 1 to point 2 (point 2 - point 1). The distance is measured in pixels and pushed onto the stack as a F26Dot6.
    If a is 0, the distance is measured in the grid fitted outline, if a is 1 its measured in the original outline.
    """
    assert a in [0,1]
    assert hasattr(I, "g"), "No glyph loaded"
    p1 = I.get_point(uint32(pop(I)), "zp1", "original" if a else "fitted")
    p2 = I.get_point(uint32(pop(I)), "zp0", "original" if a else "fitted")
    p1p = orthogonal_projection(p1, I.gs["projection_vector"])
    p2p = orthogonal_projection(p2, I.gs["projection_vector"])
    push(I, F26Dot6.to_bytes(p2p - p1p))
def MPPEM(I:"ttf.Interpreter"):
    """Measure Pixels Per EM"""
    # ignoring direction of projection vector, so scaling is linear in all directions.
    push(I, uint32.to_bytes(int((I.fontsize * I.dpi) / 72)))

"""STACK OPERATIONS"""

def push(I:"ttf.Interpreter", *x) -> None:
    if len(x) == 1 and type(x[0]) in [list, tuple]: x = tuple(x[0])
    assert all([type(i) == bytes and len(i) == 4 for i in x]), f"Cannot only push raw bytes of length 4, got {x} instead"
    assert I.sp != None, "No stack pointer (ttf.Interpreter.sp) set"
    for i in x:
        assert 0 <= I.sp < len(I.stack), f"Tried pushing outside of stack with {I.sp} in stack with length {len(I.stack)}"
        if I.debug: I.print_debug("PUSH", i)
        I.stack[I.sp] = i
        I.sp += 1
def pop(I:"ttf.Interpreter") -> Union[int, List]:
    assert I.sp != None, "No stack pointer (ttf.Interpreter.sp) set"
    I.sp -= 1
    assert len(I.stack) > I.sp >= 0, f"Tried popping from outside of stack with {I.sp}"
    if I.debug: I.print_debug("POP", I.stack[I.sp])
    return I.stack[I.sp]
def _pushB_from_program(I:"ttf.Interpreter", n):
    """Take n bytes from the instruction stream one after the other, pad them to uint32 and push onto the stack."""
    assert len(I.callstack[-1]["program"]) > I.callstack[-1]["ip"] + n
    for _ in range(n):
        I.callstack[-1]["ip"] += 1
        push(I, uint32.to_bytes(I.callstack[-1]["program"][I.callstack[-1]["ip"]]))
def _pushW_from_program(I:"ttf.Interpreter", n):
    """Take n words from the instruction stream one after the other, sign extend them to int32 and push onto the stack. The high byte of each word appears first."""
    assert len(I.callstack[-1]["program"]) > I.callstack[-1]["ip"] + n * 2
    for _ in range(n):
        v = int16(I.callstack[-1]["program"][I.callstack[-1]["ip"]+1:I.callstack[-1]["ip"] + 3])
        push(I, int32.to_bytes(v))
        I.callstack[-1]["ip"] += 2
def CINDEX(I:"ttf.Interpreter", index=None):
    """
    Copy the INDEXed element to the top of the stack
    Pop a uint32 index from the stack, copy the nth element counting from the top of the stack and push it on the top.
    """
    a = uint32(pop(I)) if index == None else index
    i = I.sp - a
    assert I.sp > i >= 0, f"invalid index: {i}"
    push(I, I.stack[i])
def MINDEX(I, index=None): # TODO: merge mindex and cindex
    """
    Move the INDEXed element to the top of the stack
    Same as CINDEX except the value is not copied up, but moved up instead.
    """
    a = index if index != None else uint32(pop(I))
    i = I.sp - a
    assert I.sp > i >= 0, f"invalid index: {i}"
    v = I.stack[i]
    for j in range(i, I.sp-1): I.stack[j] = I.stack[j+1]
    pop(I) # everything is shifted down by 1, so stack depth is reduced
    push(I, v)
def CLEAR(I:"ttf.Interpreter"): I.stack, I.sp = [None] * I.maxp.maxStackElements, 0
def DEPTH(I:"ttf.Interpreter"): push(I, uint32.to_bytes(I.sp))

"""
Storage
Writing instruction pops value first, then index. Indices must not exceed maxStorage as defined in the maxp table.
"""
def WS(I:"ttf.Interpreter"):
    """Write Store"""
    v = int32(pop(I)) # parsing as uint as the spec would demand, then rendering G in Fira Code produces insane, large values
    # if v < 0: v = -v # doing this is fine though?
    assert 0 <= (i:=uint32(pop(I))) < I.maxp.maxStorage
    I.storage[i] = v
def RS(I:"ttf.Interpreter"):
    """Read Store"""
    assert 0 <= (i:=uint32(pop(I))) < I.maxp.maxStorage
    push(I, int32.to_bytes(I.storage[i]))

"""
Control Value Table
Values in the cvt are generally expected to be in pixel units. While they technically start out in FUnits, they are supposed to be converted in the prep program. Writing in FUnits requires conversion.
Writing instructions always pop value first, then index.
Indcies must not exceed the number of total control value table elements.
"""
def WCVTF(I:"ttf.Interpreter"):
    """Write Control Value Table in Funits"""
    v = int32(pop(I)) # FUnits
    # assert v >= 0, f"{v=}"
    I.cvt[uint32(pop(I))] = F26Dot6(FU_to_px(I, v))
def WCVTP(I:"ttf.Interpreter"):
    """Write Control Value Table in Pixel units"""
    v, i = F26Dot6(pop(I)), uint32(pop(I))
    assert 0 <= i < len(I.cvt)
    I.cvt[i] = v
def RCVT(I:"ttf.Interpreter"):
    """Read Control Value Table entry"""
    assert 0 <= (i:=uint32(pop(I))) < len(I.cvt)
    push(I, F26Dot6.to_bytes(I.cvt[i]))

"""
ROUNDING
"""
def _round(I:"ttf.Interpreter", v:Union[int, float]) -> Union[int, float]:
    """Period  is difference between two neighoring rounded values. Phase is the period offset from being 0 aligned. Threshold specifies the portion that is rounded up. Since physical printing is not considered yet, engine compensation using color is not implemented."""
    if I.gs["round_state"]: 
        period, phase, threshold = I.gs["round_state"].values()
        assert 3 >= period >= 0.5 and 1.5 > phase >= 0 and 11/8 * period >= threshold >= -3/8 * period, f"{period=}, {phase=}, {threshold=}"
        v_original = v
        v -= phase
        v += threshold
        v = v // period * period
        v += phase
        if v_original >= 0 and v < 0: v = 0 + (phase if phase <= period else phase - (phase // period * period)) # lowest positive rounded value
        elif v_original < 0 and v >= 0: v = 0 - (phase if phase <= period else phase - (phase // period * period)) # highest negative rounded value
    return v
def _roundstate(I:"ttf.Interpreter", n:int, gridPeriod:int=1) -> None:
    """Decodes the last 8 bits in n to set the period, phase and threshold in roundstate"""
    match n>>6:
        case 0: period = gridPeriod/2
        case 1: period = gridPeriod
        case 2: period = gridPeriod*2
        case _: raise NotImplementedError
    match (n>>4) & 3:
        case 0: phase = 0
        case 1: phase = period / 4
        case 2: phase = period / 2
        case 3: phase = gridPeriod * 3/4
        case _: raise NotImplementedError
    match n & 0xF:
        case 0: threshold = period - 1
        case 1: threshold = -3/8 * period
        case 2: threshold = -2/8 * period
        case 3: threshold = -1/8 * period
        case 4: threshold = 0
        case 5: threshold = 1/8 * period
        case 6: threshold = 2/8 * period
        case 7: threshold = 3/8 * period
        case 8: threshold = 4/8 * period
        case 9: threshold = 5/8 * period
        case 10: threshold = 6/8 * period
        case 11: threshold = 7/8 * period
        case 12: threshold = period
        case 13: threshold = 9/8 * period
        case 14: threshold = 10/8 * period
        case 15: threshold = 11/8 * period
        case _: raise NotImplementedError
    I.gs["round_state"] = {"period": period, "phase": phase, "threshold": threshold}
def ROUND(I:"ttf.Interpreter", a): push(I, F26Dot6.to_bytes(_round(I, F26Dot6(pop(I)))))

"""GRAPHICS STATE VARIABLES"""

def SPVTL(I:"ttf.Interpreter", a):
    """Set Projection Vector To Line"""
    raise NotImplementedError
def SFVTL(I:"ttf.Interpreter", a):
    """Set Freedom Vector To Line"""
    raise NotImplementedError
def SPVFS(I:"ttf.Interpreter"):
    """Set Projection Vector From Stack"""
    y, x = EF2Dot14(pop(I)), EF2Dot14(pop(I))
    assert math.isclose(x**2 + y**2, 1, abs_tol=1e-3)
    I.gs["projection_vector"] = vec2(x,y)
def SFVFS(I:"ttf.Interpreter"): 
    """Set Freedom Vector From Stack"""
    y, x = EF2Dot14(pop(I)), EF2Dot14(pop(I))
    assert math.isclose(x**2 + y**2, 1, abs_tol=1e-3)
    I.gs["freedom_vector"] = vec2(x,y)
def GPV(I:"ttf.Interpreter"):
    """
    Get Projection Vector
    Interprets projection vector x and y components as EF2Dot14 and pushes first x, then y onto the stack.
    The values must be such that x**2 + y**2 (length of the vector) is 1.
    """
    x, y  = I.gs["projection_vector"].components()
    assert math.isclose(x**2 + y**2, 1, abs_tol=1e-3), f"Vector x**2 + y**2 must equal 1, got {x=}, {y=}, {x**2 + y**2=}."
    push(I, EF2Dot14.to_bytes(x), EF2Dot14.to_bytes(y))
def GFV(I:"ttf.Interpreter"):
    """Get Freedom Vector
    Interprets freedom vector x and y components as EF2Dot14 and pushes first x, then y onto the stack.
    The values must be such that x**2 + y**2 (length of the vector) is 1.
    """
    x, y = I.gs["freedom_vector"].components()
    assert math.isclose(x**2 + y**2, 1, abs_tol=1e-3), f"Vector x**2 + y**2 must equal 1, got {x=}, {y=}, {x**2 + y**2=}."
    push(I, EF2Dot14.to_bytes(x), EF2Dot14.to_bytes(y))
def ISECT(I:"ttf.Interpreter"): raise NotImplementedError

"""ARITHMETIC OPS"""

def ADD(I:"ttf.Interpreter"):push(I, F26Dot6.to_bytes(F26Dot6(pop(I)) + F26Dot6(pop(I))))
def SUB(I:"ttf.Interpreter"): push(I, F26Dot6.to_bytes(-F26Dot6(pop(I)) + F26Dot6(pop(I))))
def DIV(I:"ttf.Interpreter"):
    a, b = F26Dot6(pop(I)), F26Dot6(pop(I))
    if a == 0: push(I, F26Dot6.to_bytes((-1 if b < 0 else 1) * 0xFFFFFFFF))
    else: push(I, F26Dot6.to_bytes(b / a))
def MUL(I:"ttf.Interpreter"): push(I, F26Dot6.to_bytes(F26Dot6(pop(I)) * F26Dot6(pop(I))))
def ABS(I:"ttf.Interpreter"): push(I, F26Dot6.to_bytes(v if (v := F26Dot6(pop(I))) >= 0 else -v))
def NEG(I:"ttf.Interpreter"): push(I, F26Dot6.to_bytes(-F26Dot6(pop(I))))
def FLOOR(I:"ttf.Interpreter"): push(I, F26Dot6.to_bytes(math.floor(F26Dot6(pop(I)))))
def CEILING(I:"ttf.Interpreter"): push(I, F26Dot6.to_bytes(math.ceil(F26Dot6(pop(I)))))

"""BOOLEAN OPS"""

def LT(I:"ttf.Interpreter"): push(I, uint32.to_bytes(int(int32(pop(I)) > int32(pop(I)))))
def LTEQ(I:"ttf.Interpreter"): push(I, uint32.to_bytes(int(int32(pop(I)) >= int32(pop(I)))))
def GT(I:"ttf.Interpreter"): push(I, uint32.to_bytes(int(int32(pop(I)) < int32(pop(I)))))
def GTEQ(I:"ttf.Interpreter"): push(I, uint32.to_bytes(int(int32(pop(I)) <= int32(pop(I)))))
def EQ(I:"ttf.Interpreter"): push(I, uint32.to_bytes(int(int32(pop(I)) == int32(pop(I)))))
def NEQ(I:"ttf.Interpreter"): push(I, uint32.to_bytes(int(int32(pop(I)) != int32(pop(I)))))
# for ANd and OR, popping values first is necessary because python and or can sometimes evaluate after only the first value. like False and True returns False before True was looked at.
def AND(I:"ttf.Interpreter"):
    v1, v2 = uint32(pop(I)), uint32(pop(I))
    push(I, uint32.to_bytes(int(bool(v1) and bool(v2))))
def OR(I:"ttf.Interpreter"):
    v1, v2 = uint32(pop(I)), uint32(pop(I))
    push(I, uint32.to_bytes(int(bool(v1) or bool(v2))))
def NOT(I:"ttf.Interpreter"): push(I, uint32.to_bytes(int(not bool(uint32(pop(I))))))

"""FLOW CONTROL"""

def _pop_IS(I:"ttf.Interpreter", n:int=1) -> int:
    """Pops n bytes from instruction stream, advances the instruction pointer and returns the instruction at the new position. Exists to avoid duplicating error checking in _next_instruction"""
    assert type(n) == int and n > 0, f"Invalid count {n} for popping from instruction stream"
    for _ in range(n):
        assert (I.callstack[-1]["ip"] + 1) < len(I.callstack[-1]["program"]), f"Tried accessing instruction outside of range of instruction stream in {I.callstack[-1].get('context', 'program')} at {I.callstack[-1]['ip'] + 1}. Program already terminates at {len(I.callstack[-1]['program'])}."
        I.callstack[-1]["ip"] += 1
    return uint8(I.callstack[-1]["program"][I.callstack[-1]["ip"]])
def _next_instruction(I:"ttf.Interpreter") -> int:
    """Because some bytes in the instruction stream are not instructions but bytes to be pushed onto the stack, lookahead (as it is needed in IF ELSE and FDEF) must do some parsing lest what was meant to be a value is misinterpreted as an opcode.
    This function does this parsing. It reads the current instruction, skips any values associated with it and returns the next instruction while advancing the instruction pointer accordingly."""
    # NOTE: this is reimplementing some of logic of these operations and it's stupid
    name = code_name[I.callstack[-1]["program"][I.callstack[-1]["ip"]]]
    if name == "NPUSHB": _pop_IS(I, _pop_IS(I))
    elif name == "NPUSHW": _pop_IS(I, _pop_IS(I) * 2)
    elif name.startswith("PUSHB"): _pop_IS(I, int(name[-1]) + 1)
    elif name.startswith("PUSHW"): _pop_IS(I, (int(name[-1]) + 1) * 2)
    return _pop_IS(I)
def IF(I:"ttf.Interpreter"):
    """
    IF test
    Pops an integer, e, from the stack. If e is zero (FALSE), the instruction pointer is moved to the associated ELSE or EIF[] instruction in the instruction stream. If e is nonzero (TRUE), the next instruction in the instruction stream is executed. Execution continues until the associated ELSE[] instruction is encountered or the associated EIF[] instruction ends the IF[] statement. If an associated ELSE[] statement is found before the EIF[], the instruction pointer is moved to the EIF[] statement.
    """
    if not int32(pop(I)):
        lvl = 0
        op = _next_instruction(I)
        while (name:=code_name[op]) not in ["ELSE", "EIF"] or lvl != 0:
            depth = 0
            if name == "FDEF": depth += 1
            if name == "ENDF":
                assert depth > 0, "Error: ended function definition before if statement was finished."
                depth -= 1
            if name == "IF": lvl += 1
            elif name == "EIF": lvl -= 1
            op = _next_instruction(I)
        if name == "ELSE": # lookahead without moving instruction pointer to verify that there is an end to the IF statement
            # NOTE: deeply nested if statement make this run over the same shit as many times as its deep.
            original_index = I.callstack[-1]["ip"] # to reset ip later
            lvl = depth = 0
            while (name:=code_name[_next_instruction(I)]) != "EIF" or lvl !=0 :
                assert not (lvl == 0 and name == "ELSE"), f"Error: found ELSE in {I.callstack[-1]['context']} at {I.callstack[-1]['ip']} following another ELSE at {original_index}"
                if name == "FDEF": depth += 1
                elif name == "ENDF":
                    assert depth > 0, "Error: ended function definition before if statement was finished."
                    depth -= 1
                elif name == "IF": lvl += 1
                elif name == "EIF": lvl -= 1
            I.callstack[-1]["ip"] = original_index
def ELSE(I:"ttf.Interpreter"): # skip to EIF
    """
    Marks the start of the sequence of instructions that are to be executed if an IF instruction encounters a FALSE value on the stack. This sequence of instructions is terminated with an EIF instruction. ELSE is only executed directly if the IF test was TRUE and the ELSE section should be skipped. If IF test was FALSE, then this instruction is skipped.
    """
    lvl = 0
    while (name:=code_name[_next_instruction(I)]) != "EIF" or lvl != 0:
        if name == "IF": lvl += 1
        elif name == "EIF": lvl -= 1
    I.run("EIF") # for cleaner debugging only
def JROT(I:"ttf.Interpreter"):
    """Jump Relative On True"""
    e, offset = int32(pop(I)), int32(pop(I))
    if e != 0: I.callstack[-1]["ip"] += offset - 1 # -1 because ip increments automatically after this instruction
def JMPR(I:"ttf.Interpreter"):
    """
    Jump relative
    Pops an integer offset from the stack. The signed offset is added to the instruction pointer value and execution resumes from that pointer.
    """
    offset = int32(pop(I))
    assert len(I.callstack[-1]["program"]) > (ip:=(I.callstack[-1]["ip"] + offset)) >= 0
    I.callstack[-1]["ip"] = ip - 1  # after this instruction ip will increment by 1 
def CALL(I:"ttf.Interpreter", f=None, count=1):
    """Pop uint32 (i) from the stack, which is the id of a function. i must not exceed the maximum number of functions as defined in maxp table. Call that function."""
    assert I.maxp.maxFunctionDefs >= (f := f if f != None else uint32(pop(I))) >= 0, f"CALL: function index {f} outside of range 0-{I.maxp.maxFunctionDefs}"
    assert count >= 0, f"CALL: count must be bigger equal zero, got {count} instead"
    for _ in range(count): I.callstack.append(I.functions[f].copy())
def FDEF(I:"ttf.Interpreter"):
    """
    Function DEFinition
    Marks the start of a function definition. Pops a uint32 that becomes the id of the function to be defined. This id must not exceed the maximum number of functions as defined in maxp table.
    Functions can only be defined in the fpgm or prep program, not in the glyph program. Function definitions may not be nested.
    """
    assert I.maxp.maxFunctionDefs >= (f:=uint32(pop(I))) >= 0
    I.functions[f] = I.callstack[-1].copy() # before advancing instruction pointer because Interpreter.run increments it before running the instruction
    while (name:=code_name[_next_instruction(I)]) != "ENDF": assert name != "FDEF", "Error: found nested function, which are not supported"
def ENDF(I:"ttf.Interpreter"):
    """END Function definition"""
    I.callstack.pop()

"""MISC"""

def MPS(I:"ttf.Interpreter"): raise NotImplementedError
def DEBUG(I:"ttf.Interpreter"): raise NotImplementedError
def ODD(I:"ttf.Interpreter"): raise NotImplementedError
def EVEN(I:"ttf.Interpreter"): raise NotImplementedError

def DELTAP1(I:"ttf.Interpreter"): raise NotImplementedError
def DELTAP2(I:"ttf.Interpreter"): raise NotImplementedError
def DELTAP3(I:"ttf.Interpreter"): raise NotImplementedError
def DELTAC1(I:"ttf.Interpreter"): raise NotImplementedError
def DELTAC2(I:"ttf.Interpreter"): raise NotImplementedError
def DELTAC3(I:"ttf.Interpreter"): raise NotImplementedError

def JROF(I:"ttf.Interpreter"): raise NotImplementedError
def SANGW(I:"ttf.Interpreter"): raise NotImplementedError
def AA(I:"ttf.Interpreter"): raise NotImplementedError
def FLIPPT(I:"ttf.Interpreter"): raise NotImplementedError
def FLIPRGON(I:"ttf.Interpreter"): raise NotImplementedError
def FLIPRGOFF(I:"ttf.Interpreter"): raise NotImplementedError
def SCANCTRL(I:"ttf.Interpreter"):
    """Scan conversion control"""
    I.gs["scan_control"] = Euint16(pop(I))
def SDPVTL(I:"ttf.Interpreter", a): raise NotImplementedError
def GETINFO(I:"ttf.Interpreter"):
    i = uint32(pop(I))
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
    push(I, uint32.to_bytes(result))
def IDEF(I:"ttf.Interpreter"): raise NotImplementedError
def MAX(I:"ttf.Interpreter"): raise NotImplementedError #push(I, max(pop(I), pop(I)))
def MIN(I:"ttf.Interpreter"): raise NotImplementedError # push(I, min(pop(I), pop(I)))
def SCANTYPE(I:"ttf.Interpreter"): I.scantype = Euint16(pop(I))
def INSTCTRL(I:"ttf.Interpreter"):
    """Instruction execution control"""
    selector = int32(pop(I))
    v = Euint16(pop(I))
    assert selector in [1,2,3]
    if selector == 1:
        assert v in [0,1]
        I.instruction_control["stop_grid_fit"] = bool(v)
    elif selector == 2:
        assert v in [0,2]
        I.instruction_control["default_gs"] = bool(v)
    elif selector == 3: pass # ClearType related, idk
def MDRP(I:"ttf.Interpreter", a): raise NotImplementedError
def MIRP(I:"ttf.Interpreter", a): raise NotImplementedError

@dataclass(frozen=True)
class Op:
    code:int
    name:str
    func:Callable

OPS = [
    Op(0x00, "SVTCA0", lambda I: I.gs.update({"freedom_vector": vec2(0, 1), "projection_vector": vec2(0, 1)})),
    Op(0x01, "SVTCA1", lambda I: I.gs.update({"freedom_vector": vec2(1, 0), "projection_vector": vec2(1, 0)})),
    Op(0x02, "SPVTCA0", lambda I: I.gs.update({"projection_vector": vec2(0, 1)})),
    Op(0x03, "SPVTCA1", lambda I: I.gs.update({"projection_vector": vec2(1, 0)})),
    Op(0x04, "SFVTCA0", lambda I: I.gs.update({"freedom_vector": vec2(0, 1)})),
    Op(0x05, "SFVTCA1", lambda I: I.gs.update({"freedom_vector": vec2(1, 0)})),
    Op(0x06, "SPVTL0", partial(SPVTL, a=0)),
    Op(0x07, "SPVTL1", partial(SPVTL, a=1)),
    Op(0x08, "SFVTL0", partial(SFVTL, a=0)),
    Op(0x09, "SFVTL", partial(SFVTL, a=1)),
    Op(0x0A, "SPVFS", SPVFS),
    Op(0x0B, "SFVFS", SFVFS),
    Op(0x0C, "GPV", GPV),
    Op(0x0D, "GFV", GFV),
    Op(0x0E, "SFVTPV", lambda I: I.gs.update({"freedom_vector": I.gs["projection_vector"]})),
    Op(0x0F, "ISECT", ISECT),
    Op(0x10, "SRP0", lambda I: I.gs.update({"rp0": uint32(pop(I))})),
    Op(0x11, "SRP1", lambda I: I.gs.update({"rp1": uint32(pop(I))})),
    Op(0x12, "SRP2", lambda I: I.gs.update({"rp2": uint32(pop(I))})),
    Op(0x13, "SZP0", lambda I: I.gs.update({"zp0": uint32(pop(I))})),
    Op(0x14, "SZP1", lambda I: I.gs.update({"zp1": uint32(pop(I))})),
    Op(0x15, "SZP2", lambda I: I.gs.update({"zp2": uint32(pop(I))})),
    Op(0x16, "SZPS", lambda I: I.gs.update({"zp0": (v:=uint32(pop(I))), "zp1": v, "zp2": v})),
    Op(0x17, "SLOOP", lambda I: I.gs.update({"loop": int32(pop(I))})),
    Op(0x18, "RTG", lambda I: I.gs.update({"round_state": {"period": 1, "phase": 0, "threshold": 0.5}})),
    Op(0x19, "RTHG", lambda I: I.gs.update({"round_state": {"period": 1, "phase": 0.5, "threshold": 0.5}})),
    Op(0x1A, "SMD", lambda I: I.gs.update({"minimum_distance": F26Dot6(pop(I))})),
    Op(0x1B, "ELSE", ELSE),
    Op(0x1C, "JMPR", JMPR),
    Op(0x1D, "SCVTCI", lambda I: I.gs.update({"control_value_cut_in": F26Dot6(pop(I))})),
    Op(0x1E, "SSWCI", lambda I: I.gs.update({"single_width_cut_in": F26Dot6(pop(I))})),
    Op(0x1F, "SSW", lambda I: I.gs.update({"single_width_value": F26Dot6(FU_to_px(I, int32(pop(I))))})),
    Op(0x20, "DUP", partial(CINDEX, index=1)),
    Op(0x21, "POP", pop),
    Op(0x22, "CLEAR", CLEAR),
    Op(0x23, "SWAP", partial(MINDEX, index=2)),
    Op(0x24, "DEPTH", DEPTH),
    Op(0x25, "CINDEX", CINDEX),
    Op(0x26, "MINDEX", MINDEX),
    Op(0x27, "ALIGNPTS", ALIGNPTS),
    Op(0x29, "UTP", UTP),
    Op(0x2A, "LOOPCALL", lambda I: CALL(I, f=uint32(pop(I)), count=uint32(pop(I)))),
    Op(0x2B, "CALL", CALL),
    Op(0x2C, "FDEF", FDEF),
    Op(0x2D, "ENDF", ENDF),
    Op(0x2E, "MDAP0", partial(MDAP, a=0)),
    Op(0x2F, "MDAP1", partial(MDAP, a=1)),
    Op(0x30, "IUP0", partial(IUP, a=0)),
    Op(0x31, "IUP1", partial(IUP, a=1)),
    Op(0x32, "SHP0", partial(SHP, a=0)),
    Op(0x33, "SHP1", partial(SHP, a=1)),
    Op(0x34, "SHC0", partial(SHC, a=0)),
    Op(0x35, "SHC1", partial(SHC, a=1)),
    Op(0x36, "SHZ0", partial(SHZ, a=0)),
    Op(0x37, "SHZ1", partial(SHZ, a=1)),
    Op(0x38, "SHPIX", SHPIX),
    Op(0x39, "IP", IP),
    Op(0x3A, "MSIRP0", partial(MSIRP, a=0)),
    Op(0x3B, "MSIRP1", partial(MSIRP, a=1)),
    Op(0x3C, "ALIGNRP", ALIGNRP),
    Op(0x3D, "RTDG", lambda I: I.gs.update({"round_state": {"period": 0.5, "phase": 0, "threshold": 0.25}})),
    Op(0x3E, "MIAP0", partial(MIAP, a=0)),
    Op(0x3F, "MIAP1", partial(MIAP, a=1)),
    Op(0x40, "NPUSHB", lambda I: _pushB_from_program(I, _pop_IS(I))),
    Op(0x41, "NPUSHW", lambda I: _pushW_from_program(I, _pop_IS(I))),
    Op(0x42, "WS", WS),
    Op(0x43, "RS", RS),
    Op(0x44, "WCVTP", WCVTP),
    Op(0x45, "RCVT", RCVT),
    Op(0x46, "GC0", partial(GC, a=0)),
    Op(0x47, "GC1", partial(GC, a=1)),
    Op(0x48, "SCFS", SCFS),
    Op(0x49, "MD0", partial(MD, a=0)),
    Op(0x4A, "MD1", partial(MD, a=1)),
    Op(0x4B, "MPPEM", MPPEM),
    Op(0x4C, "MPS", MPS),
    Op(0x4D, "FLIPON", lambda I: I.gs.update({"auto_flip": True})),
    Op(0x4E, "FLIPOFF", lambda I: I.gs.update({"auto_flip": False})),
    Op(0x4F, "DEBUG", DEBUG),
    Op(0x50, "LT", LT),
    Op(0x51, "LTEQ", LTEQ),
    Op(0x52, "GT", GT),
    Op(0x53, "GTEQ", GTEQ),
    Op(0x54, "EQ", EQ),
    Op(0x55, "NEQ", NEQ),
    Op(0x56, "ODD", ODD),
    Op(0x57, "EVEN", EVEN),
    Op(0x58, "IF", IF),
    Op(0x59, "EIF", lambda I: None), # End IF: marks the end of an IF instruction. Nothing happens.
    Op(0x5A, "AND", AND),
    Op(0x5B, "OR", OR),
    Op(0x5C, "NOT", NOT),
    Op(0x5D, "DELTAP1", DELTAP1),
    Op(0x5E, "SDB", lambda I: I.gs.update({"delta_base": uint32(pop(I))})),
    Op(0x5F, "SDS", lambda I: I.gs.update({"delta_shift": uint32(pop(I))})),
    Op(0x60, "ADD", ADD),
    Op(0x61, "SUB", SUB),
    Op(0x62, "DIV", DIV),
    Op(0x63, "MUL", MUL),
    Op(0x64, "ABS", ABS),
    Op(0x65, "NEG", NEG),
    Op(0x66, "FLOOR", FLOOR),
    Op(0x67, "CEILING", CEILING),
    Op(0x68, "ROUND0", partial(ROUND, a=0)),
    Op(0x69, "ROUND1", partial(ROUND, a=1)),
    Op(0x6A, "ROUND2", partial(ROUND, a=2)),
    Op(0x6B, "ROUND3", partial(ROUND, a=3)),
    Op(0x6C, "NROUND0", lambda I: None), # NROUND is skipped because no engine compensation using color is implemented
    Op(0x6D, "NROUND1", lambda I: None),
    Op(0x6E, "NROUND2", lambda I: None),
    Op(0x6F, "NROUND3", lambda I: None),
    Op(0x70, "WCVTF", WCVTF),
    Op(0x71, "DELTAP2", DELTAP2),
    Op(0x72, "DELTAP3", DELTAP3),
    Op(0x73, "DELTAC1", DELTAC1),
    Op(0x74, "DELTAC2", DELTAC2),
    Op(0x75, "DELTAC3", DELTAC3),
    Op(0x76, "SROUND", lambda I: _roundstate(uint32(pop(I)), gridPeriod=1)),
    Op(0x77, "S45ROUND", lambda I: _roundstate(uint32(pop(I)), gridPeriod=math.sqrt(2) / 2)),
    Op(0x78, "JROT", JROT),
    Op(0x79, "JROF", JROF),
    Op(0x7A, "ROFF", lambda I: I.gs.update({"round_state": False})),
    Op(0x7C, "RUTG", lambda I: I.gs.update({"round_state": {"period": 1, "phase": 0, "threshold": 0.99}})),
    Op(0x7D, "RDTG", lambda I: I.gs.update({"round_state": {"period": 1, "phase": 0, "threshold": 0}})),
    Op(0x7E, "SANGW", SANGW),
    # 0x, "",F: AA, irrelevant. not even in microsoft TrueType spe)c
    Op(0x80, "FLIPPT", FLIPPT),
    Op(0x81, "FLIPRGON", FLIPRGON),
    Op(0x82, "FLIPRGOFF", FLIPRGOFF),
    Op(0x85, "SCANCTRL", SCANCTRL),
    Op(0x86, "SDPVTL0", partial(SDPVTL, a=0)),
    Op(0x87, "SDPVTL1", partial(SDPVTL, a=1)),
    Op(0x88, "GETINFO", GETINFO),
    Op(0x89, "IDEF", IDEF),
    Op(0x8A, "ROLL", partial(MINDEX, index=3)),
    Op(0x8B, "MAX", MAX),
    Op(0x8C, "MIN", MIN),
    Op(0x8D, "SCANTYPE", SCANTYPE),
    Op(0x8E, "INSTCTRL", INSTCTRL),
    Op(0xB0, "PUSHB0", lambda I: _pushB_from_program(I, 1)),
    Op(0xB1, "PUSHB1", lambda I: _pushB_from_program(I, 2)),
    Op(0xB2, "PUSHB2", lambda I: _pushB_from_program(I, 3)),
    Op(0xB3, "PUSHB3", lambda I: _pushB_from_program(I, 4)),
    Op(0xB4, "PUSHB4", lambda I: _pushB_from_program(I, 5)),
    Op(0xB5, "PUSHB5", lambda I: _pushB_from_program(I, 6)),
    Op(0xB6, "PUSHB6", lambda I: _pushB_from_program(I, 7)),
    Op(0xB7, "PUSHB7", lambda I: _pushB_from_program(I, 8)),
    Op(0xB8, "PUSHW0", lambda I: _pushW_from_program(I, 1)),
    Op(0xB9, "PUSHW1", lambda I: _pushW_from_program(I, 2)),
    Op(0xBA, "PUSHW2", lambda I: _pushW_from_program(I, 3)),
    Op(0xBB, "PUSHW3", lambda I: _pushW_from_program(I, 4)),
    Op(0xBC, "PUSHW4", lambda I: _pushW_from_program(I, 5)),
    Op(0xBD, "PUSHW5", lambda I: _pushW_from_program(I, 6)),
    Op(0xBE, "PUSHW6", lambda I: _pushW_from_program(I, 7)),
    Op(0xBF, "PUSHW7", lambda I: _pushW_from_program(I, 8)),
    Op(0xC0, "MDRP0", partial(MDRP, a=0)),
    Op(0xC1, "MDRP1", partial(MDRP, a=1)),
    Op(0xC2, "MDRP2", partial(MDRP, a=2)),
    Op(0xC3, "MDRP3", partial(MDRP, a=3)),
    Op(0xC4, "MDRP4", partial(MDRP, a=4)),
    Op(0xC5, "MDRP5", partial(MDRP, a=5)),
    Op(0xC6, "MDRP6", partial(MDRP, a=6)),
    Op(0xC7, "MDRP7", partial(MDRP, a=7)),
    Op(0xC8, "MDRP8", partial(MDRP, a=8)),
    Op(0xC9, "MDRP9", partial(MDRP, a=9)),
    Op(0xCA, "MDRP10", partial(MDRP, a=10)),
    Op(0xCB, "MDRP11", partial(MDRP, a=11)),
    Op(0xCC, "MDRP12", partial(MDRP, a=12)),
    Op(0xCD, "MDRP13", partial(MDRP, a=13)),
    Op(0xCE, "MDRP14", partial(MDRP, a=14)),
    Op(0xCF, "MDRP15", partial(MDRP, a=15)),
    Op(0xD0, "MDRP16", partial(MDRP, a=16)),
    Op(0xD1, "MDRP17", partial(MDRP, a=17)),
    Op(0xD2, "MDRP18", partial(MDRP, a=18)),
    Op(0xD3, "MDRP19", partial(MDRP, a=19)),
    Op(0xD4, "MDRP20", partial(MDRP, a=20)),
    Op(0xD5, "MDRP21", partial(MDRP, a=21)),
    Op(0xD6, "MDRP22", partial(MDRP, a=22)),
    Op(0xD7, "MDRP23", partial(MDRP, a=23)),
    Op(0xD8, "MDRP24", partial(MDRP, a=24)),
    Op(0xD9, "MDRP25", partial(MDRP, a=25)),
    Op(0xDA, "MDRP26", partial(MDRP, a=26)),
    Op(0xDB, "MDRP27", partial(MDRP, a=27)),
    Op(0xDC, "MDRP28", partial(MDRP, a=28)),
    Op(0xDD, "MDRP29", partial(MDRP, a=29)),
    Op(0xDE, "MDRP30", partial(MDRP, a=30)),
    Op(0xDF, "MDRP31", partial(MDRP, a=31)),
    Op(0xE0, "MIRP0", partial(MIRP, a=0)),
    Op(0xE1, "MIRP1", partial(MIRP, a=1)),
    Op(0xE2, "MIRP2", partial(MIRP, a=2)),
    Op(0xE3, "MIRP3", partial(MIRP, a=3)),
    Op(0xE4, "MIRP4", partial(MIRP, a=4)),
    Op(0xE5, "MIRP5", partial(MIRP, a=5)),
    Op(0xE6, "MIRP6", partial(MIRP, a=6)),
    Op(0xE7, "MIRP7", partial(MIRP, a=7)),
    Op(0xE8, "MIRP8", partial(MIRP, a=8)),
    Op(0xE9, "MIRP9", partial(MIRP, a=9)),
    Op(0xEA, "MIRP10", partial(MIRP, a=10)),
    Op(0xEB, "MIRP11", partial(MIRP, a=11)),
    Op(0xEC, "MIRP12", partial(MIRP, a=12)),
    Op(0xED, "MIRP13", partial(MIRP, a=13)),
    Op(0xEE, "MIRP14", partial(MIRP, a=14)),
    Op(0xEf, "MIRP15", partial(MIRP, a=15)),
    Op(0xF0, "MIRP16", partial(MIRP, a=16)),
    Op(0xF1, "MIRP17", partial(MIRP, a=17)),
    Op(0xF2, "MIRP18", partial(MIRP, a=18)),
    Op(0xF3, "MIRP19", partial(MIRP, a=19)),
    Op(0xF4, "MIRP20", partial(MIRP, a=20)),
    Op(0xF5, "MIRP21", partial(MIRP, a=21)),
    Op(0xF6, "MIRP22", partial(MIRP, a=22)),
    Op(0xF7, "MIRP23", partial(MIRP, a=23)),
    Op(0xF8, "MIRP24", partial(MIRP, a=24)),
    Op(0xF9, "MIRP25", partial(MIRP, a=25)),
    Op(0xFA, "MIRP26", partial(MIRP, a=26)),
    Op(0xFB, "MIRP27", partial(MIRP, a=27)),
    Op(0xFC, "MIRP28", partial(MIRP, a=28)),
    Op(0xFD, "MIRP29", partial(MIRP, a=29)),
    Op(0xFE, "MIRP30", partial(MIRP, a=30)),
    Op(0xFF, "MIRP31", partial(MIRP, a=31))
]
code_name = {op.code: op.name for op in OPS}
code_op = {op.code: op.func for op in OPS}
# name_op = {op.name: op.func for op in OPS}
name_code = {op.name: op.code for op in OPS}