import sys
import os
import datetime
import logging
import uuid
import math
from PIL import Image, ImageOps 
import cv2
import numpy as np

from .image_compare import ImageCompare
from ditherer.ditherer import Ditherer
from ditherer.dither_kernel import DitherKernel
from shapizer.triangularizer import Triangularizer
from shapizer.righttriangularizer import Righttriangularizer

#FIXME: repeated all over the place
PATH = 'out/'

class TriangleCompare(ImageCompare):

    def __init__(self, d:Ditherer, original_img:str=None, greyscale:str=None):
        super(TriangleCompare, self).__init__(d, original_img, greyscale)

    def imgtest(self, kernel:DitherKernel, testname:str, serpentine:bool, now:str=None)->np.float64:
        """Standard test for Triangular dither. Works off a .npy file of greyscale info.
        Uses original image to determine size, and the .npy file for the pixelated version
        of the image. Logs location of images it writes and returns score"""

        idd = str(uuid.uuid4())
        if not os.path.exists(f"{PATH}{now}"):
            os.makedirs(f"{PATH}{now}")
        os.mkdir(f"{PATH}{now}/{idd}")

        self.sigma = 9.0 # higher blur for large-pixel work

        logging.info(f"BEGIN {testname} TEST AT {datetime.datetime.now()}")
        logging.info(f"UUID CODE is {idd}")
        logging.info(f"OUTPUT DIR is {idd}")

        np_load_old = np.load
        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        # call load_data with allow_pickle implicitly set to true
        greyset = np.load(self.greyscale)
        # restore np.load for future normal usage
        np.load = np_load_old

        logging.info("Calling triangularizer")
        tri = Triangularizer(f"{PATH}{now}")

        logging.info("Open original image")
        image = Image.open(self.original_img).convert('LA').convert("RGBA")
        width, height = image.size

        logging.info("dither() it")
        self.d = Ditherer(greyset)
        dithered_tri = self.d.dither(kernel, serpentine)

        logging.info("Calling pixelate() WITH FILE SAVING")
        dithered_img = tri.pixelate(image, width, height, f"{PATH}{now}/{idd}/tri.png", dithered_tri, save_files=True)

        filename = f"{idd}/greyset_fs"

        image = image.convert("L")
        dithered_img = np.array(dithered_img.convert("L"))

        logging.info("Calling run_test()")
        score = self.run_test(dithered_img, np.array(image), filename, testname, serpentine)

        logging.info("export to file")
        cv2.imwrite(f"{PATH}{idd}/{filename}_{'serp' if serpentine else 'ltor'}.PNG", dithered_img)

        logging.info(f"COMPLETE {testname} TEST AT {datetime.datetime.now()}")

        return score