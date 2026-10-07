"""Scanline fill for closed loops, with every coordinate kept in subpixel integers.

One pixel holds SUBPIXELS steps on both axes, so the sample point of pixel
(column, row) sits at (column * SUBPIXELS + HALF, row * SUBPIXELS + HALF).  A
pixel is covered when its sample point falls inside the path under the chosen
fill rule.  Boundaries are half open: a sample point that lands exactly on the
right side or the bottom side of a loop counts as outside, so two loops that
share an edge never claim the same pixel.

The edges of every loop are collected into an edge table, swept row by row
through an active edge table, and the crossings of a row are paired into pixel
spans.  Everything is plain integer arithmetic; nothing here draws, reads or
prints.
"""

from functools import cmp_to_key

SUBPIXELS = 16
HALF = SUBPIXELS // 2
FILL_RULES = ("even-odd", "nonzero")


class RasterizerError(ValueError):
    """Raised for malformed loops, fill rules, clip windows or rows."""


def _floor_div(top, bottom):
    """Integer division rounding towards minus infinity; bottom is positive."""
    return top // bottom


def _ceil_div(top, bottom):
    """Integer division rounding towards plus infinity; bottom is positive."""
    return -_floor_div(-top, bottom)


def _check_int(value, what):
    if isinstance(value, bool) or not isinstance(value, int):
        raise RasterizerError("%s must be an integer, got %r" % (what, value))
    return int(value)


def _check_vertex(vertex):
    if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
        raise RasterizerError("a vertex must be a pair of integers, got %r" % (vertex,))
    return _check_int(vertex[0], "a vertex coordinate"), _check_int(
        vertex[1], "a vertex coordinate"
    )


def _check_loop(vertices):
    if not isinstance(vertices, (list, tuple)):
        raise RasterizerError("a loop must be a sequence of vertices, got %r" % (vertices,))
    points = [_check_vertex(vertex) for vertex in vertices]
    if len(points) < 3:
        raise RasterizerError("a loop needs at least three vertices, got %d" % (len(points),))
    if len(points) > 3 and points[0] == points[-1]:
        points.pop()
    return tuple(points)


def _check_path(loops):
    if not isinstance(loops, (list, tuple)):
        raise RasterizerError("a path must be a sequence of loops, got %r" % (loops,))
    path = [_check_loop(loop) for loop in loops]
    if not path:
        raise RasterizerError("a path needs at least one loop")
    return tuple(path)


def _check_clip(clip):
    if clip is None:
        return None
    if not isinstance(clip, (list, tuple)) or len(clip) != 4:
        raise RasterizerError("a clip window must be (left, top, right, bottom)")
    left = _check_int(clip[0], "a clip coordinate")
    top = _check_int(clip[1], "a clip coordinate")
    right = _check_int(clip[2], "a clip coordinate")
    bottom = _check_int(clip[3], "a clip coordinate")
    if right < left or bottom < top:
        raise RasterizerError("a clip window must not be inverted, got %r" % (clip,))
    return left, top, right, bottom


class Edge(object):
    """A non horizontal loop edge, normalised to run downwards."""

    __slots__ = ("x", "y_top", "y_bottom", "dx", "dy", "winding")

    def __init__(self, x, y_top, y_bottom, dx, dy, winding):
        self.x = x
        self.y_top = y_top
        self.y_bottom = y_bottom
        self.dx = dx
        self.dy = dy
        self.winding = winding

    def __repr__(self):
        return "Edge(x=%d, y=%d..%d, dx=%d, dy=%d, winding=%d)" % (
            self.x,
            self.y_top,
            self.y_bottom,
            self.dx,
            self.dy,
            self.winding,
        )

    def first_row(self):
        """The first pixel row whose sample line meets this edge."""
        return _ceil_div(self.y_top - HALF, SUBPIXELS)

    def crossing(self, sample):
        """Exact x where the sample line cuts the edge, as (numerator, denominator)."""
        return (self.x * self.dy + (sample - self.y_top) * self.dx, self.dy)


def _edge_from(first, second):
    """The edge between two loop vertices, or None when the edge is horizontal."""
    x0, y0 = first
    x1, y1 = second
    if y0 == y1:
        return None
    if y0 <= y1:
        x_top, y_top, x_bottom, y_bottom = x0, y0, x1, y1
        winding = 1
    else:
        x_top, y_top, x_bottom, y_bottom = x1, y1, x0, y0
        winding = -1
    return Edge(x_top, y_top, y_bottom, x_bottom - x_top, y_bottom - y_top, winding)


def _compare_crossings(left, right):
    """Order crossings by x, and a downward edge before an upward one."""
    (first, first_den), first_winding = left
    (second, second_den), second_winding = right
    left_over = first * second_den
    right_over = second * first_den
    if left_over != right_over:
        return -1 if left_over < right_over else 1
    if first_winding == second_winding:
        return 0
    return -1 if first_winding > second_winding else 1


def _parity_spans(crossings):
    """Spans between consecutive pairs of crossings under the even-odd rule."""
    spans = []
    index = 0
    while index + 1 < len(crossings):
        spans.append((crossings[index][0], crossings[index + 1][0]))
        index += 2
    return spans


def _winding_spans(crossings):
    """Spans where the winding stays away from zero under the nonzero rule."""
    spans = []
    start = None
    winding = 0
    for point, delta in crossings:
        winding += delta
        if winding == 0:
            if start is not None:
                spans.append((start, point))
                start = None
        elif start is None:
            start = point
    return spans


class ScanlineRasterizer(object):
    """Fills a path of closed loops into the pixels of a subpixel grid."""

    def __init__(self, loops, fill_rule="even-odd", clip=None):
        if fill_rule not in FILL_RULES:
            raise RasterizerError(
                "unknown fill rule %r, expected one of %r" % (fill_rule, FILL_RULES)
            )
        self._loops = _check_path(loops)
        self._fill_rule = fill_rule
        self._clip = _check_clip(clip)
        self._edge_table = self._build_edge_table()

    def __repr__(self):
        return "ScanlineRasterizer(%d loops, fill_rule=%r, clip=%r)" % (
            len(self._loops),
            self._fill_rule,
            self._clip,
        )

    @property
    def loops(self):
        """The closed loops of the path."""
        return self._loops

    @property
    def fill_rule(self):
        """The rule used to decide whether a sample point is inside."""
        return self._fill_rule

    @property
    def clip(self):
        """The inclusive pixel window the output is trimmed to, or None."""
        return self._clip

    def _build_edge_table(self):
        """Edges bucketed by the first pixel row they meet."""
        table = {}
        for loop in self._loops:
            for index, vertex in enumerate(loop):
                edge = _edge_from(vertex, loop[(index + 1) % len(loop)])
                if edge is None:
                    continue
                table.setdefault(edge.first_row(), []).append(edge)
        return table

    def _row_range(self):
        """The first and the last pixel row the path can reach."""
        if not self._edge_table:
            return 0, -1
        first = min(self._edge_table)
        last = max(
            _ceil_div(y - HALF, SUBPIXELS) - 1 for loop in self._loops for _, y in loop
        )
        return first, last

    def _columns(self, start, end, row):
        """The pixel columns of one span, or None when the span covers nothing."""
        first = _ceil_div(start[0] - HALF * start[1], SUBPIXELS * start[1])
        last = _ceil_div(end[0] - HALF * end[1], SUBPIXELS * end[1]) - 1
        if self._clip is not None:
            left, top, right, bottom = self._clip
            if row < top or row > bottom:
                return None
            first = max(first, left)
            last = min(last, right)
        if first > last:
            return None
        return first, last

    def _sweep(self):
        """Yield every pixel row of the path together with the spans it covers."""
        first, last = self._row_range()
        active = []
        for row in range(first, last + 1):
            active.extend(self._edge_table.get(row, ()))
            sample = row * SUBPIXELS + HALF
            active = [edge for edge in active if sample < edge.y_bottom]
            crossings = [(edge.crossing(sample), edge.winding) for edge in active]
            crossings.sort(key=cmp_to_key(_compare_crossings))
            if self._fill_rule == "nonzero":
                pairs = _winding_spans(crossings)
            else:
                pairs = _parity_spans(crossings)
            spans = []
            for start, end in pairs:
                columns = self._columns(start, end, row)
                if columns is not None:
                    spans.append(columns)
            yield row, spans

    def scanline(self, row):
        """Covered pixel columns of one row, as inclusive (first, last) pairs."""
        _check_int(row, "a pixel row")
        for current, spans in self._sweep():
            if current >= row:
                return spans if current == row else []
        return []

    def rasterize(self):
        """Every covered pixel as a (column, row) pair, ordered by row then column."""
        pixels = []
        for row, spans in self._sweep():
            for first, last in spans:
                pixels.extend((column, row) for column in range(first, last + 1))
        return tuple(pixels)

    def bounds(self):
        """Inclusive bounding box of the covered pixels, or None when there are none."""
        pixels = self.rasterize()
        if not pixels:
            return None
        columns = [pixel[0] for pixel in pixels]
        rows = [pixel[1] for pixel in pixels]
        return min(columns), min(rows), max(columns), max(rows)
