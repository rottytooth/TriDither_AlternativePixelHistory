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

#FIXME: repeated all over the place
PATH = 'out/'

class SquareCompare(ImageCompare):

    def __init__(self, d:Ditherer, original_img:str=None, greyscale:str=None):
        super(SquareCompare, self).__init__(d, original_img, greyscale)

    # OLD version:
    # def test_ditheralgo(self, kernel:DitherKernel, filename:str, testid:str, serpentine:bool, testname:str)->np.float64:

    def imgtest(self, kernel:DitherKernel, testname:str, serpentine:bool, now:str=None)->np.float64:
        "Dithers and then scores result; writes out image dithered, returns score"
        image = self.getImage(PATH + self.original_img)
        self.d = Ditherer(image)
        dithered_img = self.d.dither(kernel, serpentine)
        score = self.run_test(dithered_img, image, self.original_img, testname, serpentine)

        idd = str(uuid.uuid4())
        filename = f"{idd}/greyset_fs"

        cv2.imwrite(f"{PATH}{testname}/{filename}_{'serp' if serpentine else 'ltor'}.PNG", dithered_img)
        return score