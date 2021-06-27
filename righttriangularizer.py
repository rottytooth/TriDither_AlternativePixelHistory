import numpy
import math
from PIL import Image, ImageDraw
import logging
import numpy as np

from dither_kernel import DitherKernel, DITHERS
from ditherer import Ditherer
from shapizer import Shapizer


PIXEL_HEIGHT = 10 # size of triangular pixels to draw in rectangular pixels

SAVE_IMG_FILES = True


# following Globals should ONLY be used OUTSIDE of Triangularizer object
# and used to set parameters for Triangularizer

# original source image
INPUT_IMG = "source_img/gradient_700.jpg"

# the pixelated, triangular version will go here
OUT_TRI_FINAL = "out_img/gradient_rt_700.png"

# a numpy array with greyscale data for pixelated version
# this is used because pixelating the original image and averaging the color for each pixel is expensive. The Ditherer will use this to calculate dithered version of the image
GREY_LOCATION = "data/gradient_rt_700.npy"


class Righttriangularizer(Shapizer):

    def __init__(self, default_outpath:str=None):
        self.default_outpath = default_outpath
        self.greylocation = GREY_LOCATION
        self.pixel_height = PIXEL_HEIGHT

    def _get_vertices_for_max_shape(self, rwidth:int, rheight:int, pixel_unitsize:bool = False)->list:
        '''Returns the maximal vertical triangle for the available rectangle
        '''

        # widest possible triangle width that will fit in the rectangle
        # this is either the rect width, or the rect height
        triwidth = min(rwidth, rheight)

        if (pixel_unitsize): # if we need to crop in terms of a unit size
            triwidth = math.floor(triwidth / self.pixel_height) * self.pixel_height

        trianglepoints = [(0, triwidth), (triwidth, triwidth), (0, 0)]

        return trianglepoints

    def _get_max_rect(self, rwidth:int, rheight:int)->list:
        newwidth = math.floor(rwidth / self.pixel_height) * self.pixel_height
        newheight = math.floor(rheight / self.pixel_height) * self.pixel_height
        return [(0, 0),(newwidth, 0), (newwidth, newheight), (0, newheight)]

        
    def pixelate(self, im:Image.Image, width:int, height:int, outlocation:str=None, greyscale_list=None, rectangular=False)->Image.Image:
        '''
        For an entire image, make the pixels triangular, working off of a greyscale list (numpy array) if provided

        The assumption is that a greyscale_list is a dithered version of that list (everything at black or white) and so if one is provided, we save to the outdithered location.
        '''
        triwidth = self.pixel_height

        greyset = [] # all the pixel shades of grey that we will dither

        shape = self._get_max_rect(width, height) if rectangular else self._get_vertices_for_max_shape(width, height, True)

        im = self.crop_to_shape(im, shape, True)

        draw = ImageDraw.Draw(im)

        for row in range(math.floor(min(height, width) / self.pixel_height)):
            greyset.append([])
            logging.debug(f"beginning row {row}")

            width_of_row = math.floor(width / self.pixel_height) * 2 if rectangular else row * 2 + 1

            for col in range(0, width_of_row): 
                oddcol = (col % 2 == 1)
                xoffset = (col / 2 - (.5 if oddcol else 0)) * triwidth
                if oddcol:
                    #downturned
                    triangle = [
                        (xoffset, triwidth * row),
                        (xoffset + triwidth, triwidth * row),
                        (xoffset + triwidth, triwidth * (row + 1))
                    ]
                else:
                    #upturned
                    triangle = [
                        (xoffset, triwidth * row),
                        (xoffset, triwidth * (row + 1)),
                        (xoffset + triwidth, triwidth * (row + 1))
                    ]

                # find average grey and print it
                if (greyscale_list is None): # if we are in the initial pixelating step, find avg color:
                    # make an image of just that triangle pixel so we can average the color
                    small_tri = self.crop_to_shape(im, triangle)

                    # crop away the image outside that pixel
                    small_tri = small_tri.crop(self._get_rect_for_polygon(triangle))

                    color = math.floor(self.find_avg_color(small_tri))

                else:
                    color = greyscale_list[row][col]

                draw.polygon(triangle, fill=(int(color),int(color),int(color),255))
                greyset[row].append(color)

        if SAVE_IMG_FILES:
            if outlocation is None:
                outlocation = self.default_outpath
            im.save(outlocation)
            logging.info(f"Saved to path {outlocation}")

            if greyscale_list is None:
                # if we've generated a greyset, it means this is the pixelated version for a non-pixelated original, so we will save it to greylocation 
                #FIXME: it might make sense for greylocation to be overrideable like outlocation
                np.save(self.greylocation, greyset)
                logging.info(f"Saved greyset to {self.greylocation}")


        return im
    

if __name__ == '__main__':
    logFormatter = logging.Formatter("%(asctime)s  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    tri = Righttriangularizer(OUT_TRI_FINAL)

    # read image as RGB, convert to greyscale, and add alpha (transparency)
    im = Image.open(INPUT_IMG).convert('LA').convert("RGBA")

    iwidth, iheight = im.size

    # this creates the triangular-pixelled version
    iim = tri.pixelate(im, iwidth, iheight, rectangular=True)



    # Take im data returned from Triangularizer and dither it:
    # ditherer = Ditherer(greyset)
    # dith_greys = ditherer.dither(DitherKernel("FloydSteinberg"), False)

    # im = im.getdata()
    # tri.pixelate(im, iwidth, iheight, dith_greys)

