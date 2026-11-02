# Note: avoid self.subTest in these tests. VS Code's Python extension turns subtests into child items
# after the first run, then skips the parent test on every later run.
import os
import shutil
import numpy as np
from PIL import Image, ImageDraw

from shapizer.shapizer_factory import ShapizerFactory

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# where tests write their files (set TRIDITHER_TEST_OUT to change it); cleared at the start of each test run
TEST_OUT_DIR = os.path.abspath(os.environ.get("TRIDITHER_TEST_OUT", os.path.join(PROJECT_DIR, "out_test")))
TEST_OUT_MARKER = ".tridither_test_out" # only a folder holding this marker (or an empty one) is ever cleared

SHAPES = ["triangle", "righttriangle", "hexagon", "hexagon_rev"]

_test_out_cleared = False

def clear_test_out():
    "Empty TEST_OUT_DIR, once per test run (the first time a test asks for a folder)"
    global _test_out_cleared
    if _test_out_cleared:
        return
    if os.path.isdir(TEST_OUT_DIR) and os.listdir(TEST_OUT_DIR) and not os.path.exists(os.path.join(TEST_OUT_DIR, TEST_OUT_MARKER)):
        raise RuntimeError(f"{TEST_OUT_DIR} has files the tests didn't create; not clearing it")
    shutil.rmtree(TEST_OUT_DIR, ignore_errors=True)
    os.makedirs(TEST_OUT_DIR, exist_ok=True)
    open(os.path.join(TEST_OUT_DIR, TEST_OUT_MARKER), "w").close()
    _test_out_cleared = True

def make_layout(testcase) -> str:
    """The code uses paths relative to src/ (../out, ../data, ...), so build that layout in this test's folder under TEST_OUT_DIR.
    Tests don't chdir into it (VS Code's test runner reports results relative to the cwd)"""
    clear_test_out()
    root = os.path.join(TEST_OUT_DIR, ".".join(testcase.id().split(".")[-3:])) # e.g. test_cli.CliTest.test_pixelate_dither_score
    for folder in ["src", "source_img", "pixelated_img", "out_img", "data", "out", "logs"]:
        os.makedirs(os.path.join(root, folder), exist_ok=True)
    return root

def make_test_image(path:str, width:int=120, height:int=100):
    "Horizontal gradient with a dark circle in the middle"
    im = Image.new("L", (width, height))
    im.putdata([int(255 * x / (width - 1)) for y in range(height) for x in range(width)])
    ImageDraw.Draw(im).ellipse((width // 4, height // 4, 3 * width // 4, 3 * height // 4), fill=40)
    im.save(path)

def pixelate(root:str, shape:str, source:str):
    "Same steps as the 'pixelate' process in __main__.py; returns (pixelated png path, greyset npy path)"
    png = os.path.join(root, "pixelated_img", f"{shape}.png")
    npy = os.path.join(root, "data", f"{shape}.npy")
    shapizer = ShapizerFactory.GetShapizer(shape)
    shapizer.greylocation = npy
    with Image.open(source).convert('LA').convert("RGBA") as image:
        width, height = image.size
        shapizer.pixelate(image, width, height, png, rectangular=(shape == 'righttriangle'), save_files=True)
    return png, npy

def load_greyset(path:str) -> np.ndarray:
    return np.load(path, allow_pickle=True)

def ragged_triangle(rows:int, value:int) -> np.ndarray:
    "A greyset shaped like the Triangularizer's: row r has 2r + 1 cells"
    greyset = np.empty(rows, dtype=object)
    greyset[:] = [[value] * (2 * r + 1) for r in range(rows)]
    return greyset
