from PIL import Image, ImageOps 
import numpy as np
import cv2

from dither_kernel import DITHERS, DitherKernel
from ditherer import Ditherer
from triangularizer import Triangularizer
from righttriangularizer import Righttriangularizer
from hexagonizer import Hexagonizer

# used only to correctly size the final image
ORIGINAL_IMAGE = "source_img/david_tiny.jpg"

GREYSET = "data/david_hex_tiny.npy"

OUTPATH = f"out_img/david_tiny_hex_%PERCENT%.png"

DITHER = DitherKernel("StephensonArce")
#DITHER = DitherKernel("FloydSteinberg")
#DITHER = DitherKernel("FSTriTwo")

#tri = Righttriangularizer(None)
#tri = Triangularizer(None)
tri = Hexagonizer(None)

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

IS_EQ_TRIANGLE = False

for i in range(100,101): #range(1, 101, 3):
    
    dithered_tri = ditherer.dither(kernel=DITHER, serpentine=False, percent=i, rowoffset=IS_EQ_TRIANGLE)

    tri.pixelate(image, width, height, OUTPATH.replace("%PERCENT%",str(i)), dithered_tri) #, rectangular=True)

