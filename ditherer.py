import numpy as np
import math
import cv2
import json

from dither_kernel import DITHERS, DitherKernel

class Ditherer(object):
    def __init__(self, imgarray:np.ndarray=None):
        self.imgarray = imgarray

    #FIXME:this ought to handle three- or more-color dithering
    def dither(self, kernel:DitherKernel, serpentine:bool) -> np.ndarray:
        """
        This accepts either an image (as PNG) or a raw .npy dataset produced by pixelating a non-rectangular image. Either way, it will treat each memory cell as a pixel and dither from black to white (this is two-color only)
        """
        if self.imgarray is None:
            raise ValueError("Ditherer image not set")
            
        retarray = self.imgarray.copy()
        for y in range(0, len(self.imgarray)):

            if (serpentine == False or y % 2 == 0): # direction
                for x in range(0, len(self.imgarray[y])):
                    retarray = self.calculate_cell(kernel, x, y, retarray, True)
            else:
                for x in range(len(self.imgarray[y]) - 1, -1, -1):
                    retarray = self.calculate_cell(kernel, x, y, retarray, False)

        return retarray

    def calculate_cell(self, kernel:DitherKernel, x:int, y:int, retarray:np.ndarray, ltor:bool):

        if retarray[y][x] > 127:
            diff = 255 - retarray[y][x]
            retarray[y][x] = 255
        else:
            diff = 0 - retarray[y][x]
            retarray[y][x] = 0

        if diff == 0: return retarray

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

    PATH = '/Users/danieltemkin/Documents/TriDithers/MachineLearning/'

    img = cv2.imread(PATH + "dtemkin.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dith = Ditherer(img)

    imgresult = dith.dither(DitherKernel("FloydSteinberg"), False)
    cv2.imwrite(PATH + "dtemkin_FS.png", imgresult)

    # imgresult = dith.dither(DitherKernel("Floyd-Steinberg"), True)
    # cv2.imwrite(PATH + "David_tiny_FS_serp.png", imgresult)

    imgresult = dith.dither(DitherKernel("Atkinson"), False)
    cv2.imwrite(PATH + "dtemkin_Atkinson.png", imgresult)

    # imgresult = dith.dither(DitherKernel("Atkinson"), True)
    # cv2.imwrite(PATH + "David_tiny_Atkinson_serp.png", imgresult)

    # imgresult = dith.dither(DitherKernel("Crisp"), True)
    # cv2.imwrite(PATH + "David_tiny.png", imgresult)
