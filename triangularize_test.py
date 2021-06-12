from PIL import Image, ImageOps 
import numpy as np

from dither_kernel import DITHERS, DitherKernel
from ditherer import Ditherer
from triangularizer import Triangularizer

ORIGINAL_IMAGE = 'source_img/david_tri_big.jpg'

GREYSET = 'data/david_tri.npy'

OUTPATH = 'out_img/david_tri_big.png'

DITHER = DitherKernel("FloydSteinberg")

tri = Triangularizer(None)

np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# call load_data with allow_pickle implicitly set to true
greyset = np.load(GREYSET)
# restore np.load for future normal usage
np.load = np_load_old

image = Image.open(ORIGINAL_IMAGE).convert('LA').convert("RGBA")
width, height = image.size

ditherer = Ditherer(greyset)

dithered_tri = ditherer.dither(DITHER, False)
tri.pixelate(image, width, height, OUTPATH, dithered_tri)

