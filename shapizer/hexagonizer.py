import numpy
import math
from PIL import Image, ImageDraw
import logging
import numpy as np

from ditherer.dither_kernel import DitherKernel, DITHERS
from ditherer.ditherer import Ditherer
from .shapizer import Shapizer


OFFSET = math.sqrt(3)/2 # relationship of triangle width to height
PIXEL_HEIGHT = 20 # size of hexagonal pixels to draw in rectangular pixels

SIDE_TO_HEIGHT = 1.732051


class Hexagonizer(Shapizer):

    def __init__(self, default_outpath:str=None):
        self.default_outpath = default_outpath
        self.greylocation = None
        self.pixel_height = PIXEL_HEIGHT

    # FIXME: should we be cropping to hex? If so, this is where we do it
    # (based on Triangularizer)
    # def _get_vertices_for_max_shape(self, rwidth:int, rheight:int, pixel_unitsize:bool = False)->list:
        
    def pixelate(self, im:Image.Image, width:int, height:int, outlocation:str=None, greyscale_list=None, greyoutlocation=None, rectangular=False, save_files=False)->Image.Image:
        '''
        For an entire image, make the pixels hexagonal, working off of a greyscale list (numpy array) if provided

        The assumption is that a greyscale_list is a dithered version of that list (everything at black or white) and so if one is provided, we save to the outdithered location.
        '''
        sidelen = self.pixel_height / SIDE_TO_HEIGHT

        if greyoutlocation == None:
            greyoutlocation = self.greylocation

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

        if save_files:
            super().save(outlocation, im, greyscale_list, greyset, greyoutlocation)

        return im
    