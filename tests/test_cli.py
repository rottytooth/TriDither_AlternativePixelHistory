import os
import re
import subprocess
import sys
import unittest

from tests.helpers import SRC_DIR, make_layout, make_test_image


class CliTest(unittest.TestCase):
    "Runs the pixelate -> dither -> score processes from __main__.py for each shape"

    def setUp(self):
        self.root = make_layout(self)
        make_test_image(os.path.join(self.root, "source_img", "test.png"))

    def run_process(self, *args) -> str:
        result = subprocess.run([sys.executable, os.path.join(SRC_DIR, "__main__.py"), *args],
                                cwd=os.path.join(self.root, "src"), capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stderr # logging goes to stderr

    def pixelate_dither_score(self, shape:str):
        self.run_process("pixelate", "-i", "test.png", "-o", f"{shape}.png", "-g", f"{shape}.npy", "-s", shape)
        self.assertTrue(os.path.exists(os.path.join(self.root, "pixelated_img", f"{shape}.png")))
        self.assertTrue(os.path.exists(os.path.join(self.root, "data", f"{shape}.npy")))

        self.run_process("dither", "-i", "test.png", "-o", f"{shape}_FS.png", "-g", f"{shape}.npy", "-s", shape, "-k", "FloydSteinberg")
        self.assertTrue(os.path.exists(os.path.join(self.root, "out_img", f"{shape}_FS.png")))

        log = self.run_process("score", "-i", f"{shape}.png", "-o", f"{shape}_FS.png", "-g", f"{shape}.npy", "-s", shape)
        match = re.search(r"score = ([\d.]+)", log)
        assert match, log
        score = float(match.group(1))
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)

    def test_triangle(self):
        self.pixelate_dither_score("triangle")

    def test_righttriangle(self):
        self.pixelate_dither_score("righttriangle")

    def test_hexagon(self):
        self.pixelate_dither_score("hexagon")

    def test_hexagon_rev(self):
        self.pixelate_dither_score("hexagon_rev")


if __name__ == '__main__':
    unittest.main()
