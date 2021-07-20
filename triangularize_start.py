from PIL import Image, ImageOps 
import numpy as np
import cv2
import logging
import math

from ditherer.dither_kernel import DITHERS, DitherKernel
from ditherer.ditherer import Ditherer
from shapizer.triangularizer import Triangularizer
from shapizer.righttriangularizer import Righttriangularizer
from shapizer.hexagonizer import Hexagonizer



# following Globals should ONLY be used OUTSIDE of Triangularizer object
# and used to set parameters for Triangularizer

# original source image
INPUT_IMG = "source_img/sanding_big.jpg"

# the pixelated, triangular version will go here
OUT_TRI_FINAL = "out_img/sanding_large_tri.png"

# a numpy array with greyscale data for pixelated version
# this is used because pixelating the original image and averaging the color for each pixel is expensive. The Ditherer will use this to calculate dithered version of the image
GREY_LOCATION = "data/sanding_large_tri.npy"


if __name__ == '__main__':
    logFormatter = logging.Formatter("%(asctime)s  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    tri = Triangularizer(OUT_TRI_FINAL)

    # read image as RGB, convert to greyscale, and add alpha (transparency)
    im = Image.open(INPUT_IMG).convert('LA').convert("RGBA")

    iwidth, iheight = im.size

    # this creates the triangular-pixelled version
    logging.info(f"Begin pixelation")
    iim = tri.pixelate(im, iwidth, iheight, greyoutlocation=GREY_LOCATION)
    logging.info("Complete pixelation")


    # Take im data returned from Triangularizer and dither it:
    # ditherer = Ditherer(greyset)
    # dith_greys = ditherer.dither(DitherKernel("FloydSteinberg"), False)

    # im = im.getdata()
    # tri.pixelate(im, iwidth, iheight, dith_greys)

