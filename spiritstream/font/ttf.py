from typing import List, Union, Dict, Tuple
from spiritstream.vec import vec2
from spiritstream.dtype import *
from spiritstream.font.table import *

# Architecture notes:
# - have a ttf file object that manages table data access efficiently and lazily
def FU_to_px(I:"Interpreter", v:Union[float, int]) -> float: return v * (I.fontsize * I.dpi) / (72 * I.head.unitsPerEM)

# CHECKSUM
def pad4(b:bytes) -> bytes: return b.ljust((len(b) + 3) // 4 * 4, b'\0')
def checkSum(b:bytes) -> int: return sum([int.from_bytes(b[i:i+4], "big") for i in range(0, len(b), 4) if (b:=pad4(b))]) & 0xFFFFFFFF

def quadratic_equation(a, b, c) -> Union[None, Tuple[float]]:
    if a == 0: return None if b == 0 else (-c / b,) # not quadratic, but linear. Function should still return x for 0
    if (root:=b**2-4*a*c) < 0: return None
    x1 = (-b + math.sqrt(root)) / (2*a)
    x2 = (-b - math.sqrt(root)) / (2*a)
    return (x1,) if x1 == x2 else (x1, x2)

# used for compound glyphs
def transform(v:vec2, xscale, yscale, scale01, scale10) -> vec2: return vec2(xscale * v.x + scale10 * v.y, scale01 * v.x + yscale*v.y)

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
        self.cmap = p.parse(cmap_table, offset=self.table_directory["cmap"].offset, length=self.table_directory["cmap"].length)
        self.loca = p.parse(uint32 if self.head.indexToLocFormat else uint16, count=self.maxp.numGlyphs, offset=self.table_directory["loca"].offset)
        self.hhea = p.parse(hhea_table, offset=self.table_directory["hhea"].offset)
        self.hmtx = p.parse(hmtx_table, offset=self.table_directory["hmtx"].offset, numOfLongHorMetrics=self.hhea.numOfLongHorMetrics, numGlyphs=self.maxp.numGlyphs)
        self.glyf = p.b[(t:=self.table_directory["glyf"]).offset:t.offset+t.length] # raw bytes of glyf table
        if self.debug:
            print(f"Contains tables: {list(self.table_directory.keys())}")
            assert all((cs:=self.checksum(p.b))), f"Checksum failed: {cs}"
        del p

        self.fontsize = self.dpi = None # TODO: get value from screen device

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
    
    def getGlyph(self, unicode:int=None, fontsize=None, dpi=None, glyphIndex=None, antialiasing=0):
        self.antialiasing = antialiasing
        assert (unicode != None and isinstance(unicode, int)) or glyphIndex != None
        assert fontsize != None or self.fontsize != None, "Specify fontsize"
        assert dpi != None or self.dpi != None, "Specify dpi"
        if dpi == None: dpi = self.dpi
        else: self.dpi = dpi
        if fontsize == None: fontsize = self.fontsize
        else: self.fontsize = fontsize

        glyphIndex = glyphIndex if glyphIndex != None else self.cmap.subtable.getGlyphIndex(unicode)
        return self.scan_convert(self.hint(glyphIndex))

    def hint(self, glyphIndex, is_child=False) -> glyf:
        glyph = glyf(self.glyf[self.loca[glyphIndex]:self.loca[glyphIndex+1]]) if glyphIndex + 1 < len(self.loca) else glyf(self.glyf[self.loca[glyphIndex]:])
        if hasattr(glyph, "children"): # compound glyph
            for child in glyph.children: # child is glyphComponent object
                child_glyph = self.hint(child.glyphIndex, is_child=True)
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
                pts = [transform(vec2(x,y), xscale, yscale, scale01, scale10) for x,y in zip(child_glyph.x, child_glyph.y)]
                child_glyph.x = [v.x for v in pts]
                child_glyph.y = [v.y for v in pts]
                if child.ARGS_ARE_XY_VALUES:
                    if child.SCALED_COMPONENT_OFFSET == child.UNSCALED_COMPONENT_OFFSET: child.UNSCALED_COMPONENT_OFFSET = True # default behaviour if flags are invalid
                    if child.UNSCALED_COMPONENT_OFFSET: offset = vec2(child.arg1, child.arg2)
                    else: offset = transform(vec2(child.arg1, child.arg2), xscale, yscale, scale01, scale10)
                    offset = vec2(FU_to_px(self, child.arg1), FU_to_px(self, child.arg2))
                    if child.ROUND_XY_TO_GRID: offset = vec2((offset.x + 0.5)//1, (offset.y + 0.5) // 1) # round to grid
                else: offset = glyph.get_point(child.arg1) - child_glyph.get_point(child.arg2)
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
        else: glyph.x, glyph.y = list(map(lambda v: [FU_to_px(self, v0) for v0 in v], [glyph.x, glyph.y])) # simple glyph: convert Funits to pixels
        if not (hasattr(glyph, "advanceWidth") and hasattr(glyph, "leftSideBearing")):
            if glyphIndex < self.hhea.numOfLongHorMetrics:
                glyph.advanceWidth = FU_to_px(self, self.hmtx.longHorMetric[glyphIndex].advanceWidth)
                glyph.leftSideBearing = FU_to_px(self, self.hmtx.longHorMetric[glyphIndex].leftSideBearing)
            else:
                glyph.advanceWidth = FU_to_px(self, self.hmtx.longHorMetric[-1].advanceWidth)
                glyph.leftSideBearing = FU_to_px(self, self.hmtx.leftSideBearing[glyphIndex - self.hhea.numOfLongHorMetrics])
        
        if is_child == False and self.head.flags & 2: glyph.x = [x - glyph.leftSideBearing for x in glyph.x] # this flag means left side bearing should be aligned with x = 0. only applies when not dealing with compound glyph components
        
        return glyph

    def scan_convert(self, g:glyf) -> List[List[int]]:
        """
        Returns a 2D list of rendered pixels (Row-major) of the current glyph in g.
        Uses the non-zero winding number rule, which means: for each pixel, go right and on each intersection with any contour segment, determine the gradient at that point. if the gradient points up, add 1, else sub 1. If the result is zero, the point is outside, else, its inside
        """
        pts = [g.get_point(i) for i in range(g.endPtsContours[-1] + 1)] if g.endPtsContours != [] else [glyf.glyphPoint(0, 0, False)] # default 0 to return a bitmap of 0
        assert isinstance(self.antialiasing, int)
        xs, ys = [p.x for p in pts], [p.y for p in pts]
        minX, maxX = math.floor(min(xs)), math.ceil(max(xs))
        minY, maxY = math.floor(min(ys)), math.ceil(max(ys))
        g.bearing = vec2(math.floor(g.leftSideBearing) + minX, maxY)
        width = maxX - minX
        height = maxY - minY
        if self.antialiasing > 1:
            width, height, maxX, minX, maxY, minY = map(lambda x: x*self.antialiasing, [width, height, maxX, minX, maxY, minY])
            pts = [glyf.glyphPoint(p.x*self.antialiasing, p.y*self.antialiasing, p.onCurve) for p in pts]
        pts = [glyf.glyphPoint(p.x-minX, p.y-minY, p.onCurve) for p in pts] # shift into positive range
        bitmap = [[None for x in range(width)] for y in range(height)]
        # add oncurve points between consecutive control points
        all_contours = [] # first point will exist twice. at 0 and -1
        start = 0
        for ep in g.endPtsContours:
            this_contour = []
            contour = pts[start:ep+1]
            for p1, p2 in zip(contour, contour[1:] + [contour[0]]):
                this_contour.append(p1)
                if p1.onCurve == p2.onCurve: this_contour.append(g.glyphPoint(p1.x + (p2.x-p1.x)/2, p1.y + (p2.y-p1.y) / 2, True)) # add another point inbetween. if two consecutive offcurve points, add an oncurve one to keep the curves quadratic. if two oncurve ones, add one so the segments are always 3 points: easy to handle.
            start = ep+1
            this_contour.append(contour[0])
            assert len(this_contour) % 2 == 1
            all_contours.append(this_contour)
        for row in range(height):
            y = row + 0.5
            intersections = []
            for contour in all_contours:
                for i in range(0, len(contour)-2, 2):
                    p0, p1, p2 = contour[i], contour[i+1], contour[i+2] # p1 = control point
                    if all(p.y < y for p in (p0, p1, p2)) or all(p.y > y for p in (p0, p1, p2)): continue # segment not near this row
                    # NOTE: in both linear and bezier curves, if an intersection falls on the exact start of the segment, then that intersection is ignored. It is instead counted as an intersections with the end of the previous segment.
                    if p1.onCurve: # not an actual control point but one added to keep segments having 3 points each. this line actually goes straight from p0 to p2
                        if p2.x == p0.x:
                            if y != p0.y or [isect["x"] for isect in intersections if math.isclose(isect["x"], x) and isect["winding_number"] == (-1 if p2.y-p0.y < 0 else 1)] == []: intersections.append({"x": p2.x, "winding_number": -1 if p2.y - p0.y < 0 else 1})
                        else: # y = mx + b
                            m = (p2.y - p0.y) / (p2.x - p0.x) # account for direction of contour too
                            if m == 0: continue
                            direction_on_contour = m * (-1 if p2.x - p0.x < 0 else 1)
                            b = p2.y - (m * p2.x)
                            x = (y - b) / m
                            if not math.isclose(x, p0.x, rel_tol=1e-9) and not (math.isclose(y, p0.y, rel_tol=1e-9)) or [isect["x"] for isect in intersections if math.isclose(isect["x"], x) and isect["winding_number"] == (-1 if direction_on_contour < 0 else 1)] == []: intersections.append({"x": x, "winding_number": -1 if direction_on_contour < 0 else 1, "source": "linear"})
                    else: # actual quadratic bezier curve.
                        p0, p1, p2 = map(lambda p: vec2(p.x, p.y), (p0, p1, p2)) # convert to vectors
                        y0, y1, y2 = map(lambda p: p.y-y, (p0, p1, p2)) # shifted down so I can pretend that the x I'm interested in is at y = 0, which is necessary for using quadratic equation
                        ts = quadratic_equation(y0 - 2*y1 + y2, 2*(y1 - y0), y0)
                        ts = [t for t in ts if 1 >= t >= 0] if ts != None else []
                        for t in ts:
                            p = (((1-t)**2)*p0 + 2*(1-t)*t*p1 + (t**2)*p2)
                            x = p.x
                            gradient = 2*(1-t) * (p1-p0) + 2*t*(p2 - p1)
                            gradient = (-1 if gradient.x < 0 else 1) * (gradient / gradient.x).y if gradient.x != 0 else gradient.y # normalize x, account for zero division. if zero division, just use y to know which direction its going in, roughly. If both gradient.x and .y are 0, its a flat line and won't be counted.
                            if [isect["x"] for isect in intersections if math.isclose(isect["x"], x) and isect["winding_number"] == (-1 if gradient < 0 else 1)] == []:
                                if gradient != 0: intersections.append({"x": x, "winding_number": -1 if gradient < 0 else 1, "source": "bezier"})
            for column in range(width):
                x = column + 0.5
                relevant_intersections = [i for i in intersections if i["x"] >= x]
                bitmap[height - row - 1][column] = 1 if sum([i["winding_number"] for i in relevant_intersections]) != 0 else 0
        if self.antialiasing > 1:
            new_bitmap = [[None for x in range(width // self.antialiasing)] for y in range(height // self.antialiasing)]
            for y in range(0, height, self.antialiasing):
                for x in range(0, width, self.antialiasing):
                    values = [bitmap[ny][nx] for ny in range(y, y+self.antialiasing) for nx in range(x,x+self.antialiasing)]
                    nx, ny = int(x // self.antialiasing), int(y // self.antialiasing)
                    new_bitmap[ny][nx] = sum(values) / len(values)
            bitmap = new_bitmap
        g.bitmap = bitmap
        return g