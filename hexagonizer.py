import numpy
import math
from PIL import Image, ImageDraw
import logging
import numpy as np

from dither_kernel import DitherKernel, DITHERS
from ditherer import Ditherer
from shapizer import Shapizer


OFFSET = math.sqrt(3)/2 # relationship of triangle width to height
PIXEL_HEIGHT = 20 # size of hexagonal pixels to draw in rectangular pixels

SIDE_TO_HEIGHT = 1.732051

SAVE_IMG_FILES = True


# following Globals should ONLY be used OUTSIDE of Triangularizer object
# and used to set parameters for Triangularizer

# original source image
INPUT_IMG = "source_img/gradient_med.png"

# the pixelated, hexagonal version will go here
OUT_TRI_FINAL = "out_img/gradient_hex_med.png"

# a numpy array with greyscale data for pixelated version
# this is used because pixelating the original image and averaging the color for each pixel is expensive. The Ditherer will use this to calculate dithered version of the image
GREY_LOCATION = "data/gradient_hex_med.npy"


class Hexagonizer(Shapizer):

    def __init__(self, default_outpath:str=None):
        self.default_outpath = default_outpath
        self.greylocation = GREY_LOCATION
        self.pixel_height = PIXEL_HEIGHT

    # def _get_vertices_for_max_shape(self, rwidth:int, rheight:int, pixel_unitsize:bool = False)->list:
    #     '''Returns the maximal vertical triangle for the available rectangle
    #     '''

    #     # widest possible triangle width that will fit in the rectangle
    #     # this is either the rect width, or the rect height divided by offset,
    #     # as triangle width * offset == triangle height
    #     triwidth = min(rwidth, round(rheight / OFFSET))

    #     if (pixel_unitsize): # if we need to crop in terms of a unit size
    #         triwidth = math.floor(triwidth / self.pixel_height) * self.pixel_height
    #     trianglepoints = [(rwidth / 2 - triwidth / 2, triwidth * OFFSET), (rwidth / 2 + triwidth / 2, triwidth * OFFSET), (rwidth / 2, 0)]

    #     return trianglepoints
        
    def pixelate(self, im:Image.Image, width:int, height:int, outlocation:str=None, greyscale_list=None)->Image.Image:
        '''
        For an entire image, make the pixels triangular, working off of a greyscale list (numpy array) if provided

        The assumption is that a greyscale_list is a dithered version of that list (everything at black or white) and so if one is provided, we save to the outdithered location.
        '''
        sidelen = self.pixel_height / SIDE_TO_HEIGHT

        greyset = [] # all the pixel shades of grey that we will dither

        # im = self.crop_to_shape(im, self._get_vertices_for_max_shape(width, height, True), True)

        draw = ImageDraw.Draw(im)

        for row in range(math.ceil((height * 2) / self.pixel_height) + 2):
            greyset.append([]) # new row
            logging.debug(f"beginning row {row}")
            for col in range(math.ceil(width / (sidelen * 3)) + 2): 

                starting_x = col * sidelen * 3
                starting_y = row * self.pixel_height / 2

                if row % 2 == 1:
                    starting_x += -1.5 * sidelen

                hexagon = [
                    (starting_x, starting_y),
                    (starting_x + sidelen, starting_y),
                    (starting_x + 3/2 * sidelen, starting_y + self.pixel_height / 2),
                    (starting_x + sidelen, starting_y + self.pixel_height),
                    (starting_x, starting_y + self.pixel_height),
                    (starting_x - 1/2 * sidelen, starting_y + self.pixel_height / 2)
                ]

                # find average grey and print it
                if (greyscale_list is None): # if we are in the initial pixelating step, find avg color:
                    # make an image of just that triangle pixel so we can average the color
                    small_hex = self.crop_to_shape(im, hexagon)

                    # crop away the image outside that pixel
                    small_hex = small_hex.crop(self._get_rect_for_polygon(hexagon))

                    color = math.floor(self.find_avg_color(small_hex))

                else:
                    color = greyscale_list[row][col]

                draw.polygon(hexagon, fill=(int(color),int(color),int(color),255))
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

    tri = Hexagonizer(OUT_TRI_FINAL)

    # read image as RGB, convert to greyscale, and add alpha (transparency)
    im = Image.open(INPUT_IMG).convert('LA').convert("RGBA")

    iwidth, iheight = im.size

    # this creates the hexagon-pixelled version
    logging.info(f"Begin pixelation")
    iim = tri.pixelate(im, iwidth, iheight)
    logging.info("Complete pixelation")



