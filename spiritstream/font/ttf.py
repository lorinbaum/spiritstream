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

@dataclass(frozen=True)
class Intersection:
    x:float
    wn:int # winding number

# Quadratic eq solver with tolerance
quadratic_equation = lambda a,b,c: [-c / b] if (aok:=abs(a) < 1e-12) and abs(b) > 1e-12 else ([] if aok else (
    ([] if (d:=b*b-4*a*c)<0 else 
    [(-b - math.sqrt(d)) / (2 * a), (-b + math.sqrt(d)) / (2 * a)])))

def rasterize(contours: List[List[CurvePoint]], aa: int) -> Tuple[List[List[int]], vec2]:
    pts = [p for c in contours for p in c]
    xs, ys = [p.x for p in pts], [p.y for p in pts]
    mx, MX, my, MY = math.floor(min(xs)), math.ceil(max(xs)), math.floor(min(ys)), math.ceil(max(ys))
    sx, sy = MX - mx, MY - my
    bmp = [[0]*sx for _ in range(sy)]
    
    for row in range(sy):
        all_ints = []
        for aay in range(aa):
            y = row + (aay + .5)/aa
            ints = []
            for c in contours:
                for i in range(0, len(c)-2, 2):
                    p0, p1, p2 = [vec2(pt.x, pt.y) for pt in c[i:i+3]]
                    if all(p.y<y for p in (p0,p1,p2)) or all(p.y>y for p in (p0,p1,p2)): continue
                    a, b, cc = p0.y - 2*p1.y + p2.y, 2*(p1.y-p0.y), p0.y - y
                    ts = [t for t in quadratic_equation(a,b,cc) if -1e-9<=t<=1+1e-9]
                    for t in ts:
                        t = max(0,min(1,t))
                        x = (1-t)**2*p0.x + 2*(1-t)*t*p1.x + t**2*p2.x
                        grad = vec2(2*(1-t)*(p1.x-p0.x)+2*t*(p2.x-p1.x),
                                    2*(1-t)*(p1.y-p0.y)+2*t*(p2.y-p1.y))
                        if abs(grad.y)<1e-12: continue
                        wn = 1 if grad.y>0 else -1
                        if not any(abs(ii.x-x)<1e-9 and ii.wn==wn for ii in ints): ints.append(Intersection(x,wn))
            all_ints.append(ints)
        
        for col in range(sx):
            v=0
            for aax in range(aa):
                x = col + (aax+.5)/aa
                for ints in all_ints:
                    v += 1 if sum(i.wn for i in ints if i.x>=x) != 0 else 0
            bmp[row][col] = int(v/(aa*aa)*255)
    return bmp, vec2(sx,sy)

