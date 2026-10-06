"""Tests for the scanline rasterizer (unittest, standard library only)."""

import unittest

from rasterizer.core import RasterizerError, SUBPIXELS, ScanlineRasterizer

HALF = SUBPIXELS // 2


def px(count):
    """A length of count pixels, written in subpixel steps."""
    return count * SUBPIXELS


def half(count):
    """A position of count and a half pixels, written in subpixel steps."""
    return count * SUBPIXELS + HALF


def box(left, top, right, bottom):
    """Corners of an axis aligned box, in subpixel steps."""
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def coverage(raster):
    """Covered columns per pixel row, as {row: [column, ...]}."""
    covered = {}
    for column, row in raster.rasterize():
        covered.setdefault(row, []).append(column)
    return {row: sorted(columns) for row, columns in sorted(covered.items())}


def every(rows, columns):
    """A coverage map with the same columns on each of the given rows."""
    return {row: list(columns) for row in rows}


class BoxTests(unittest.TestCase):
    def test_a_pixel_aligned_box_fills_every_pixel_it_covers(self):
        raster = ScanlineRasterizer([box(0, 0, px(4), px(4))])
        self.assertEqual(coverage(raster), every(range(4), range(4)))
        self.assertEqual(raster.bounds(), (0, 0, 3, 3))
        self.assertEqual(len(raster.rasterize()), 16)

    def test_a_box_offset_by_half_a_pixel_keeps_its_columns(self):
        raster = ScanlineRasterizer([box(HALF, HALF, half(4), half(4))])
        self.assertEqual(coverage(raster), every(range(4), range(4)))
        self.assertEqual(raster.scanline(1), [(0, 3)])
        self.assertEqual(raster.bounds(), (0, 0, 3, 3))


class SharedEdgeTests(unittest.TestCase):
    def test_two_boxes_that_share_an_edge_never_claim_the_same_pixel(self):
        raster = ScanlineRasterizer(
            [box(HALF, HALF, half(4), half(4)), box(half(4), HALF, half(8), half(4))]
        )
        self.assertEqual(coverage(raster), every(range(4), range(8)))
        self.assertEqual(raster.bounds(), (0, 0, 7, 3))
        pixels = list(raster.rasterize())
        self.assertEqual(len(pixels), len(set(pixels)))
        self.assertEqual(len(pixels), 32)


class FillRuleTests(unittest.TestCase):
    def test_the_even_odd_rule_leaves_the_overlap_of_two_loops_empty(self):
        raster = ScanlineRasterizer(
            [box(0, 0, px(6), px(6)), box(px(3), px(3), px(9), px(9))],
            fill_rule="even-odd",
        )
        expected = {
            0: [0, 1, 2, 3, 4, 5],
            1: [0, 1, 2, 3, 4, 5],
            2: [0, 1, 2, 3, 4, 5],
            3: [0, 1, 2, 6, 7, 8],
            4: [0, 1, 2, 6, 7, 8],
            5: [0, 1, 2, 6, 7, 8],
            6: [3, 4, 5, 6, 7, 8],
            7: [3, 4, 5, 6, 7, 8],
            8: [3, 4, 5, 6, 7, 8],
        }
        self.assertEqual(coverage(raster), expected)
        self.assertEqual(raster.bounds(), (0, 0, 8, 8))
        self.assertEqual(len(raster.rasterize()), 54)

    def test_the_nonzero_rule_keeps_a_reversed_loop_as_a_hole(self):
        outer = box(0, 0, px(8), px(8))
        hole = list(reversed(box(px(3), px(3), px(5), px(5))))
        raster = ScanlineRasterizer([outer, hole], fill_rule="nonzero")
        expected = every([0, 1, 2, 5, 6, 7], range(8))
        expected[3] = [0, 1, 2, 5, 6, 7]
        expected[4] = [0, 1, 2, 5, 6, 7]
        self.assertEqual(coverage(raster), expected)
        self.assertEqual(raster.bounds(), (0, 0, 7, 7))
        self.assertEqual(len(raster.rasterize()), 60)


class BoundaryTests(unittest.TestCase):
    def test_a_horizontal_step_on_a_sample_row_does_not_spill(self):
        step = [(0, 8), (px(8), 8), (px(8), px(10) + HALF), (px(4), px(10) + HALF),
                (px(4), px(5) + HALF), (0, px(5) + HALF)]
        raster = ScanlineRasterizer([step])
        expected = every(range(5), range(8))
        expected[5] = [4, 5, 6, 7]
        expected[6] = [4, 5, 6, 7]
        expected[7] = [4, 5, 6, 7]
        expected[8] = [4, 5, 6, 7]
        expected[9] = [4, 5, 6, 7]
        self.assertEqual(coverage(raster), expected)
        self.assertEqual(raster.bounds(), (0, 0, 7, 9))
        self.assertEqual(raster.scanline(5), [(4, 7)])

    def test_a_top_vertex_between_two_sample_rows_does_not_add_a_row(self):
        raster = ScanlineRasterizer([box(0, HALF + 1, px(4), px(6) + HALF + 1)])
        self.assertEqual(coverage(raster), every(range(1, 7), range(4)))
        self.assertEqual(raster.bounds(), (0, 1, 3, 6))
        self.assertEqual(len(raster.rasterize()), 24)

    def test_a_box_that_ends_on_a_sample_row_keeps_its_bottom_row(self):
        raster = ScanlineRasterizer([box(0, HALF, px(4), px(2) + HALF)])
        self.assertEqual(coverage(raster), every(range(2), range(4)))
        self.assertEqual(raster.bounds(), (0, 0, 3, 1))
        self.assertEqual(len(raster.rasterize()), 8)


class ClipTests(unittest.TestCase):
    def test_a_clip_window_keeps_only_the_pixels_inside_it(self):
        raster = ScanlineRasterizer([box(0, 0, px(10), px(10))], clip=(1, 1, 2, 2))
        self.assertEqual(coverage(raster), {1: [1, 2], 2: [1, 2]})
        self.assertEqual(raster.bounds(), (1, 1, 2, 2))

        away = ScanlineRasterizer([box(0, 0, px(4), px(4))], clip=(8, 8, 9, 9))
        self.assertEqual(away.rasterize(), ())
        self.assertIsNone(away.bounds())

        edge = ScanlineRasterizer([box(0, 0, px(4), px(4))], clip=(3, 3, 6, 6))
        self.assertEqual(coverage(edge), {3: [3]})


class InputTests(unittest.TestCase):
    def test_malformed_input_is_rejected(self):
        with self.assertRaises(RasterizerError):
            ScanlineRasterizer([[(0, 0), (px(1), 0)]])
        with self.assertRaises(RasterizerError):
            ScanlineRasterizer([])
        with self.assertRaises(RasterizerError):
            ScanlineRasterizer([box(0, 0, px(1), px(1))], fill_rule="both")
        with self.assertRaises(RasterizerError):
            ScanlineRasterizer([box(0, 0, px(1), px(1))], clip=(4, 4, 1, 1))
        with self.assertRaises(RasterizerError):
            ScanlineRasterizer([box(0, 0, px(1), px(1))], clip=(0, 0, 4))
        with self.assertRaises(RasterizerError):
            ScanlineRasterizer([[(HALF, 0.0), (px(1), 0), (px(1), px(1))]])
        with self.assertRaises(RasterizerError):
            ScanlineRasterizer([[(0, True), (px(1), 0), (px(1), px(1))]])

        raster = ScanlineRasterizer([box(0, 0, px(1), px(1))])
        with self.assertRaises(RasterizerError):
            raster.scanline("0")
        with self.assertRaises(RasterizerError):
            raster.scanline(1.5)
        self.assertEqual(raster.scanline(-1), [])
        self.assertEqual(raster.scanline(9), [])

        flat = ScanlineRasterizer([[(0, 0), (px(5), 0), (px(10), 0)]])
        self.assertEqual(flat.rasterize(), ())
        self.assertIsNone(flat.bounds())
        self.assertEqual(flat.scanline(0), [])


if __name__ == "__main__":
    unittest.main()
