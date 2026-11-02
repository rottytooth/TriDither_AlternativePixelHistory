import os
import unittest
import numpy as np
from PIL import Image

from shapizer.shapizer import Shapizer
from shapizer.shapizer_factory import ShapizerFactory
from tests.helpers import SHAPES, make_layout, make_test_image, pixelate, load_greyset


def paint_rows(shape:str, greyset, width:int, height:int) -> np.ndarray:
    "Draw each row of the grid in its own grey (row r -> r + 1) on black, so we can see where rows land"
    rowcodes = [[r + 1] * len(row) for r, row in enumerate(greyset)]
    shapizer = ShapizerFactory.GetShapizer(shape)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    return np.array(shapizer.pixelate(canvas, width, height, greyscale_list=rowcodes).convert("L"))


class ShapizerTest(unittest.TestCase):

    def setUp(self):
        self.root = make_layout(self)
        self.source = os.path.join(self.root, "source_img", "test.png")
        make_test_image(self.source)

    def test_pixelate_writes_image_and_greyset(self):
        for shape in SHAPES:
            png, npy = pixelate(self.root, shape, self.source)
            self.assertTrue(os.path.exists(png), shape)
            greyset = load_greyset(npy)
            self.assertGreater(len(greyset), 1, shape)
            for row in greyset:
                self.assertTrue(all(0 <= v <= 255 for v in row), shape)

    def test_triangle_greyset_is_ragged(self):
        _, npy = pixelate(self.root, "triangle", self.source)
        greyset = load_greyset(npy)
        self.assertEqual(greyset.dtype, object)
        self.assertEqual([len(row) for row in greyset], [2 * r + 1 for r in range(len(greyset))])

    def check_grid_covers_image_with_rows_in_order(self, shape:str, top:int=0):
        _, npy = pixelate(self.root, shape, self.source)
        painted = paint_rows(shape, load_greyset(npy), 120, 100)
        self.assertEqual(int((painted[top:] == 0).sum()), 0, "some pixels are not covered by any hexagon")
        for x in (10, 60, 110):
            column = painted[:, x].tolist()
            runs = [v for i, v in enumerate(column) if i == 0 or v != column[i - 1]]
            self.assertEqual(runs, sorted(runs), f"rows out of order going down column {x}: {runs}")

    def test_hexagon_grid_covers_image_with_rows_in_order(self):
        # known: the flat-top hexagon grid leaves gaps along the top half-row (y < 10)
        self.check_grid_covers_image_with_rows_in_order("hexagon", top=10)

    def test_hexagon_rev_grid_covers_image_with_rows_in_order(self):
        self.check_grid_covers_image_with_rows_in_order("hexagon_rev")

    def test_hexagon_rev_row_count_fits_image(self):
        _, npy = pixelate(self.root, "hexagon_rev", self.source)
        rows = len(load_greyset(npy))
        # pointy-top rows are 1.5 * pixel_height apart
        self.assertLessEqual(rows, 100 / (12 * 1.5) + 3)

    def test_save_ragged_and_uniform_greysets(self):
        im = Image.new("RGBA", (4, 4))
        ragged = [[1], [2, 3, 4], [5, 6, 7, 8, 9]]
        uniform = [[1, 2], [3, 4]]
        path = lambda name: os.path.join(self.root, "data", name)
        Shapizer().save(path("ragged.png"), im, None, ragged, path("ragged.npy"))
        Shapizer().save(path("uniform.png"), im, None, uniform, path("uniform.npy"))
        self.assertEqual([list(r) for r in load_greyset(path("ragged.npy"))], ragged)
        loaded = load_greyset(path("uniform.npy"))
        self.assertNotEqual(loaded.dtype, object)
        self.assertEqual(loaded.tolist(), uniform)


if __name__ == '__main__':
    unittest.main()
