from PIL import Image, ImageOps 
import numpy as np
import cv2

from dither_kernel import DITHERS, DitherKernel
from ditherer import Ditherer
from triangularizer import Triangularizer
from righttriangularizer import Righttriangularizer

ORIGINAL_IMAGE = "source_img/gradient_medsmall.png"

GREYSET = "data/gradient_tri_medsmall.npy"

OUTPATH = f"out_img/gradient_tri_medsmall_FS_%PERCENT%.png"

DITHER = DitherKernel("FloydSteinberg")

#tri = Righttriangularizer(None)
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

for i in range(1, 100, 3):
    
    dithered_tri = ditherer.dither(DITHER, False, i)

    tri.pixelate(image, width, height, OUTPATH.replace("%PERCENT%",str(i)), dithered_tri)

