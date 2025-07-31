from typing import List, Union, Dict, Tuple
from spiritstream.vec import vec2
from spiritstream.dtype import *
from spiritstream.font.table import *
from spiritstream.font.font import Glyph
from dataclasses import dataclass

# Architecture notes:
# - have a ttf file object that manages table data access efficiently and lazily

@dataclass(frozen=True)
class CurvePoint:
    x:float
    y:float
    onCurve:bool

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

class TTF:
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
    
    def fupx(self, v:Union[float, int]) -> float: return v * (self.fontsize * self.dpi) / (72 * self.head.unitsPerEM) # Font Unit to pixel conversion
    
    # hilariously inefficient
    def glyph(self, unicode, fontsize, dpi):
        self.fontsize = fontsize
        self.dpi = dpi
        g = self.loadglyph(self.cmap.subtable.getGlyphIndex(unicode))
        if g.x:
            minX, maxX = math.floor(min(g.x)), math.ceil(max(g.x))
            minY, maxY = math.floor(min(g.y)), math.ceil(max(g.y))
            size = vec2(maxX - minX, maxY - minY)
            bearing = vec2(g.leftSideBearing - (min(g.x) - minX), maxY)
        else:
            size = vec2(0, 0)
            bearing = g.leftSideBearing
        return Glyph(size, bearing, g.advanceWidth)

    def render(self, unicode:int=None, fontsize=None, dpi=None, glyphIndex=None, antialiasing=0):
        self.antialiasing = antialiasing
        assert (unicode != None and isinstance(unicode, int)) or glyphIndex != None
        assert fontsize != None or self.fontsize != None, "Specify fontsize"
        assert dpi != None or self.dpi != None, "Specify dpi"
        if dpi == None: dpi = self.dpi
        else: self.dpi = dpi
        if fontsize == None: fontsize = self.fontsize
        else: self.fontsize = fontsize

        glyphIndex = glyphIndex if glyphIndex != None else self.cmap.subtable.getGlyphIndex(unicode)
        return self.rasterize(self.loadglyph(glyphIndex))

    # @functools.cache
    def loadglyph(self, glyphIndex, is_child=False) -> glyf:
        glyph = glyf(self.glyf[self.loca[glyphIndex]:self.loca[glyphIndex+1]]) if glyphIndex + 1 < len(self.loca) else glyf(self.glyf[self.loca[glyphIndex]:])
        if hasattr(glyph, "children"): # compound glyph
            for child in glyph.children: # child is glyphComponent object
                child_glyph = self.loadglyph(child.glyphIndex, is_child=True)
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
                    offset = vec2(self.fupx(child.arg1), self.fupx(child.arg2))
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
        else: glyph.x, glyph.y = list(map(lambda v: [self.fupx(v0) for v0 in v], [glyph.x, glyph.y])) # simple glyph: convert Funits to pixels
        if not (hasattr(glyph, "advanceWidth") and hasattr(glyph, "leftSideBearing")):
            if glyphIndex < self.hhea.numOfLongHorMetrics:
                glyph.advanceWidth = self.fupx(self.hmtx.longHorMetric[glyphIndex].advanceWidth)
                glyph.leftSideBearing = self.fupx(self.hmtx.longHorMetric[glyphIndex].leftSideBearing)
            else:
                glyph.advanceWidth = self.fupx(self.hmtx.longHorMetric[-1].advanceWidth)
                glyph.leftSideBearing = self.fupx(self.hmtx.leftSideBearing[glyphIndex - self.hhea.numOfLongHorMetrics])
        
        if is_child == False and self.head.flags & 2: glyph.x = [x - glyph.leftSideBearing for x in glyph.x] # this flag means left side bearing should be aligned with x = 0. only applies when not dealing with compound glyph components
        return glyph

    def rasterize(self, g:glyf) -> Glyph:
        """
        Returns a 2D list of rendered pixels (Row-major) of the current glyph in g.
        Uses the non-zero winding number rule, which means: for each pixel, go right and on each intersection with any contour segment, determine the gradient at that point. if the gradient points up, add 1, else sub 1. If the result is zero, the point is outside, else, its inside

        because rasteriztation adds some space to the outside, it should subtract that space from bearing.x (less to move in when rendering because already moved in) and add it to bearing.y (glyph gets taller)
        """
        assert len(g.x) == len(g.y) and isinstance(self.antialiasing, int) and self.antialiasing >= 1
        if len(g.x) == 0: return []
        minX, maxX = math.floor(min(g.x)), math.ceil(max(g.x))
        minY, maxY = math.floor(min(g.y)), math.ceil(max(g.y))
        g.bearing = vec2(g.leftSideBearing - (min(g.x) - minX), maxY)
        pts = [CurvePoint(x-minX, y-minY, bool(f & 0x01)) for x,y,f in zip(g.x, g.y, g.flags)] # shift into positive range
        # add oncurve points between consecutive control points
        all_contours = [] # first point will exist twice. at 0 and -1
        start = 0
        for ep in g.endPtsContours:
            new_contour = []
            contour = pts[start:ep+1]
            for p1, p2 in zip(contour, contour[1:] + [contour[0]]):
                new_contour.append(p1)
                # add another point inbetween. if two consecutive offcurve points, add an oncurve one to keep the curves quadratic. if two oncurve ones, add one so the segments are always 3 points: easy to handle.
                if p1.onCurve == p2.onCurve: new_contour.append(CurvePoint(p1.x + (p2.x-p1.x)/2, p1.y + (p2.y-p1.y) / 2, True)) 
            start = ep+1
            new_contour.append(contour[0])
            assert len(new_contour) % 2 == 1
            all_contours.append(new_contour)
        bitmap, size = rasterize(all_contours, self.antialiasing)
        return bitmap
    
def rasterize(contours:List[List[CurvePoint]], antialiasing:int) -> Tuple[List[List[float]], vec2]:
        assert isinstance(antialiasing, int) and antialiasing >= 1
        pts = [p for contour in contours for p in contour]
        xs, ys = [p.x for p in pts], [p.y for p in pts]
        minX, maxX = math.floor(min(xs)), math.ceil(max(xs))
        minY, maxY = math.floor(min(ys)), math.ceil(max(ys))
        size = vec2(maxX-minX, maxY-minY)
        bitmap = [[None for x in range(size.x)] for y in range(size.y)]

        for row in range(size.y):
            all_intersections = []
            for aay in range(antialiasing):
                y = row + aay * 1 / antialiasing - 1 / antialiasing / 2
                intersections = []
                for contour in contours:
                    for i in range(0, len(contour)-2, 2):
                        p0, p1, p2 = contour[i], contour[i+1], contour[i+2] # p1 = control point
                        if all(p.y < y for p in (p0, p1, p2)) or all(p.y > y for p in (p0, p1, p2)): continue # segment not near this row
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
                                if t in [0, 1]:
                                    if p1.y > p0.y: winding_number = -1
                                    elif p1.y < p0.y: winding_number = 1
                                    else:
                                        if p2.y > p0.y: winding_number = -1
                                        elif p2.y < p0.y: winding_number = 1
                                        else: continue
                                else:
                                    gradient = 2*(1-t) * (p1-p0) + 2*t*(p2 - p1)
                                    gradient = (-1 if gradient.x < 0 else 1) * (gradient / gradient.x).y if gradient.x != 0 else gradient.y # normalize x, account for zero division. if zero division, just use y to know which direction its going in, roughly. If both gradient.x and .y are 0, its a flat line and won't be counted.
                                    winding_number = -1 if gradient < 0 else 1
                                intersections.append({"x": x, "winding_number": winding_number})
                                
                all_intersections.append(intersections)
            for column in range(size.x):
                v = 0
                for aax in range(antialiasing):
                    x = column + aax * 1 / antialiasing - 1 / antialiasing / 2
                    for intersections in all_intersections:
                        v += 1 if sum([i["winding_number"] for i in intersections if i["x"] >= x]) != 0 else 0
                bitmap[size.y - row - 1][column] = int(v / antialiasing**2 * 255)
        bitmap.reverse()
        return bitmap, size
