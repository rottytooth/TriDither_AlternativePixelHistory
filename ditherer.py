import numpy as np
import math
import cv2
import json
from PIL import Image

from dither_kernel import DITHERS, DitherKernel

class Ditherer(object):
    def __init__(self, imgarray:np.ndarray=None):
        self.imgarray = imgarray

    #FIXME:this ought to handle three- or more-color dithering
    def dither(self, kernel:DitherKernel, serpentine:bool=False, percent:int=100) -> np.ndarray:
        """
        This accepts either an image (as PNG) or a raw .npy dataset produced by pixelating a non-rectangular image. Either way, it will treat each memory cell as a pixel and dither from black to white (this is two-color only)
        """
        if self.imgarray is None:
            raise ValueError("Ditherer image not set")
            
        retarray = self.imgarray.copy()

        if percent < 100:
            retarray = self.change_contrast(retarray, percent)

        for y in range(0, len(self.imgarray)):

            if (serpentine == False or y % 2 == 0): # direction
                for x in range(0, len(self.imgarray[y])):
                    retarray = self.calculate_cell(kernel, x, y, retarray, True, percent)
            else:
                for x in range(len(self.imgarray[y]) -1, -1, -1):
                    retarray = self.calculate_cell(kernel, x, y, retarray, False, percent)

        return retarray
    
    def change_contrast(self, img_arr:np.ndarray, percent:int):

        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img_arr)
        img.load()

        if not (0 < percent <= 100):
            raise "out of range"
        level = 259 - (2.59 * percent)
        factor = (259 * (level + 255)) / (255 * (259 - level))
        def contrast(c):
            value = 128 + factor * (c - 128)
            return max(0, min(255, value))
        final_img = img.point(contrast)

        return cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2GRAY)

    def calculate_cell(self, kernel:DitherKernel, x:int, y:int, retarray:np.ndarray, ltor:bool, percent:int):

        original_value = retarray[y][x]

        if retarray[y][x] > 127:
            diff = 255 - retarray[y][x]
            retarray[y][x] = 255
        else:
            diff = 0 - retarray[y][x]
            retarray[y][x] = 0

        if diff == 0: return retarray

        p = percent / 100.0 * 255

        for j in range(0, len(kernel.nums)):
            for i in range(0, len(kernel.nums[j])):
                x_use = x + i - kernel.start_x
                y_use = y + j - kernel.start_y

                if not ltor:
                    # we're flowing right to left and need to flip the location of the
                    # x we're using
                    x_use = x - i + kernel.start_x

                # if we are not out of range
                if y_use >= 0 and y_use < len(self.imgarray) \
                    and x_use >= 0 and x_use < len(self.imgarray[y]):

                    retarray[y_use][x_use] = self.clamp(retarray[y_use][x_use] - diff * kernel.nums[j][i] / kernel.denom, 0, 255)

        return retarray

    def clamp(self, val, minv, maxv):
        return max(min(maxv, val), minv)

#-----------------------------------------------

if __name__ == '__main__':

    '''
        This just does a straight dither -- by default, of individual square pixels
    '''

    PATH = '/Users/danieltemkin/Repos/TriDither_AlternativePixelHistory/out_img/'

    IN_PATH = '/Users/danieltemkin/Repos/TriDither_AlternativePixelHistory/source_img/'



    percent = 50


    # open in PIL and then pass to Ditherer as cv2   
    # img_to_contrast = Image.open(IN_PATH + "gradient.png")
    # img_to_contrast.load()
    # img = change_contrast2(img_to_contrast, percent)
    # img_cv2 = np.array(img)
    # dith = Ditherer(cv2.cvtColor(img_cv2, cv2.COLOR_RGB2GRAY))

    # work off image already in Ditherer
    dith = Ditherer(cv2.cvtColor(cv2.imread(IN_PATH + "gradient.png"), cv2.COLOR_BGR2GRAY))


    imgresult = dith.dither(DitherKernel("FloydSteinberg"), False, percent)
    cv2.imwrite(PATH + f"gradient_FS_{percent}ci-init.png", imgresult)

    # imgresult = dith.dither(DitherKernel("Floyd-Steinberg"), True)
    # cv2.imwrite(PATH + "David_tiny_FS_serp.png", imgresult)

    imgresult = dith.dither(DitherKernel("Atkinson"), False, percent)
    cv2.imwrite(PATH + f"gradient_Atkinson_{percent}ci-init.png", imgresult)

    # imgresult = dith.dither(DitherKernel("Atkinson"), True)
    # cv2.imwrite(PATH + "David_tiny_Atkinson_serp.png", imgresult)

    # imgresult = dith.dither(DitherKernel("Crisp"), True)
    # cv2.imwrite(PATH + "David_tiny.png", imgresult)
