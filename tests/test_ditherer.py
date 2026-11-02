import os
import unittest
import numpy as np

from ditherer.ditherer import Ditherer
from ditherer.dither_kernel import DitherKernel
from tests.helpers import DATA_DIR, load_greyset, ragged_triangle

# kernels that diffuse all of their error (Atkinson deliberately drops 2/8, so it's left out)
FULL_KERNELS = ["FloydSteinberg", "Jarvis, Judice and Ninke", "Stucki", "Burkes", "Sierra", "Two-Row Sierra"]


class DithererTest(unittest.TestCase):

    def test_output_is_black_and_white_and_keeps_dtype(self):
        for dtype in (np.uint8, np.int64):
            out = Ditherer(np.full((40, 40), 100, dtype=dtype)).dither(DitherKernel("FloydSteinberg"))
            self.assertEqual(out.dtype, dtype)
            self.assertTrue(set(np.unique(out).tolist()) <= {0, 255}, dtype.__name__)

    def test_mean_grey_is_preserved(self):
        # catches truncated error diffusion and uint8 wraparound on dark pixels
        failures = []
        for name in FULL_KERNELS:
            for value in (10, 64, 128, 200, 245):
                for dtype in (np.uint8, np.int64):
                    mean = Ditherer(np.full((64, 64), value, dtype=dtype)).dither(DitherKernel(name)).mean()
                    if abs(mean - value) > 4:
                        failures.append(f"{name}, {value}, {dtype.__name__}: mean {mean:.1f}")
        self.assertEqual(failures, [])

    def test_input_is_not_modified(self):
        img = np.full((20, 20), 90, dtype=np.uint8)
        Ditherer(img).dither(DitherKernel("FloydSteinberg"))
        self.assertTrue((img == 90).all())

    def test_ragged_triangle_greyset(self):
        out = Ditherer(ragged_triangle(40, 64)).dither(DitherKernel("FloydSteinberg"), rowoffset=True)
        values = np.concatenate([np.array(row, dtype=float) for row in out])
        self.assertEqual(len(out), 40)
        self.assertEqual([len(row) for row in out], [2 * r + 1 for r in range(40)])
        self.assertTrue(set(values.tolist()) <= {0, 255})
        self.assertAlmostEqual(values.mean(), 64, delta=8) # more error falls off the slanted edges

    def test_down_kernel_is_used_for_odd_columns(self):
        img = np.full((10, 10), 100, dtype=np.uint8)
        same = Ditherer(img).dither(DitherKernel("FloydSteinberg"), DitherKernel("FloydSteinberg"))
        different = Ditherer(img).dither(DitherKernel("FloydSteinberg"), DitherKernel("Stucki"))
        np.testing.assert_array_equal(same, Ditherer(img).dither(DitherKernel("FloydSteinberg")))
        self.assertFalse(np.array_equal(same, different))


@unittest.skipUnless(os.path.exists(os.path.join(DATA_DIR, "david_tri.npy")), "project data folder not present")
class ExistingDataTest(unittest.TestCase):
    "Greysets saved by older numpy versions still load and dither"

    def test_triangle_greyset(self):
        greyset = load_greyset(os.path.join(DATA_DIR, "david_tri.npy"))
        self.assertEqual(greyset.dtype, object)
        self.assertEqual([len(row) for row in greyset], [2 * r + 1 for r in range(len(greyset))])
        out = Ditherer(greyset).dither(DitherKernel("FloydSteinberg"), rowoffset=True)
        self.assertTrue(set(np.concatenate([np.array(r, dtype=float) for r in out]).tolist()) <= {0, 255})

    def test_hexagon_greyset(self):
        greyset = load_greyset(os.path.join(DATA_DIR, "david_hex.npy"))
        self.assertEqual(greyset.ndim, 2)
        out = Ditherer(greyset[:60, :60]).dither(DitherKernel("FloydSteinberg"))
        self.assertTrue(set(np.unique(out).tolist()) <= {0, 255})


if __name__ == '__main__':
    unittest.main()
