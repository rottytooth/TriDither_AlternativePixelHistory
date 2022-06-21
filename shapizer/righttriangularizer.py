import numpy
import math
from PIL import Image, ImageDraw
import logging
import numpy as np

from ditherer.dither_kernel import DitherKernel, DITHERS
from ditherer.ditherer import Ditherer
from shapizer.shapizer import Shapizer


PIXEL_HEIGHT = 10 # size of triangular pixels to draw in rectangular pixels


class Righttriangularizer(Shapizer):

    def __init__(self, default_outpath:str=None):
        self.default_outpath = default_outpath
        self.greylocation = None
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
        
    def pixelate(self, im:Image.Image, width:int, height:int, outlocation:str=None, greyscale_list=None, greyoutlocation=None, rectangular=False, save_files=False)->Image.Image:
        '''
        For an entire image, make the pixels triangular, working off of a greyscale list (numpy array) if provided

        The assumption is that a greyscale_list is a dithered version of that list (everything at black or white) and so if one is provided, we save to the outdithered location.
        '''

        if greyoutlocation == None:
            greyoutlocation = self.greylocation

        triwidth = self.pixel_height

        greyset = [] # all the pixel shades of grey that we will dither

        shape = self._get_max_rect(width, height) if rectangular else self._get_vertices_for_max_shape(width, height, True)

        im = self.crop_to_shape(im, shape, True)

        draw = ImageDraw.Draw(im)

        rows_to_draw = math.floor(height / self.pixel_height)
        if not rectangular: # if we are going triangular, we need square aspect ratio
            rows_to_draw = math.floor(min(height, width) / self.pixel_height)

        for row in range(rows_to_draw):
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

        if save_files:
            super().save(outlocation, im, greyscale_list, greyset, greyoutlocation)

        return im
    