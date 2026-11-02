import os
import unittest
from unittest import mock
import numpy as np

from ditherer.ditherer import Ditherer
from ditherer.dither_kernel import DitherKernel
from tester import image_compare
from tester.image_compare import ImageCompare, BLUR_LEVEL
from tests.helpers import make_layout, make_test_image, pixelate

real_dither = Ditherer.dither


class ScoreImageTest(unittest.TestCase):

    def setUp(self):
        self.imc = ImageCompare(Ditherer())
        self.imc.sigma = BLUR_LEVEL
        x = np.linspace(0, 255, 200)
        self.gradient = np.tile(x, (200, 1)).astype(np.uint8)

    def test_identical_images_score_one(self):
        self.assertAlmostEqual(self.imc.scoreImage(self.gradient, self.gradient).ssim, 1.0, places=6)

    def test_flat_grey_vs_its_dither_scores_high(self):
        # with the wrong data_range, leftover dither texture in flat areas scores near 0
        flat = np.full((200, 200), 64, dtype=np.uint8)
        dithered = Ditherer(flat).dither(DitherKernel("FloydSteinberg"))
        self.assertGreater(self.imc.scoreImage(flat, dithered).ssim, 0.9)

    def test_dither_beats_noise_and_threshold(self):
        dithered = Ditherer(self.gradient).dither(DitherKernel("FloydSteinberg"))
        noise = (np.random.default_rng(0).integers(0, 2, self.gradient.shape) * 255).astype(np.uint8)
        threshold = np.where(self.gradient > 127, 255, 0).astype(np.uint8)
        score = self.imc.scoreImage(self.gradient, dithered).ssim
        self.assertGreater(score, self.imc.scoreImage(self.gradient, noise).ssim)
        self.assertGreater(score, self.imc.scoreImage(self.gradient, threshold).ssim)


class ImgtestOtherTest(unittest.TestCase):
    "The scoring path used by the genetic algorithm and exhaustive runs"

    def setUp(self):
        self.root = make_layout(self)
        self.source = os.path.join(self.root, "source_img", "test.png")
        make_test_image(self.source)
        out_dir = mock.patch.object(image_compare, "OUT_DIR", os.path.join(self.root, "out"))
        out_dir.start()
        self.addCleanup(out_dir.stop)

    def run_imgtest(self, shape):
        png, npy = pixelate(self.root, shape, self.source)
        imc = ImageCompare(Ditherer(), png, npy)
        with mock.patch.object(Ditherer, "dither", autospec=True, side_effect=real_dither) as dither:
            score, idd = imc.imgtest(DitherKernel("FloydSteinberg"), "genetic", False, "testrun", 1, shape)
        return score, idd, dither

    def test_triangle_uses_row_offset(self):
        score, _, dither = self.run_imgtest("triangle")
        self.assertTrue(dither.call_args.kwargs["rowoffset"])
        self.assertGreater(score, 0.5)

    def test_other_shapes_do_not_use_row_offset(self):
        for shape in ["righttriangle", "hexagon", "hexagon_rev"]:
            score, idd, dither = self.run_imgtest(shape)
            self.assertFalse(dither.call_args.kwargs["rowoffset"], shape)
            self.assertGreater(score, 0.5, shape)
            self.assertTrue(os.path.exists(os.path.join(self.root, "out", "testrun", "1", idd, f"{shape}1.png")), shape)


if __name__ == '__main__':
    unittest.main()
