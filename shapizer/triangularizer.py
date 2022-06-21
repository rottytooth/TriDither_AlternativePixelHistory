import numpy
import math
from PIL import Image, ImageDraw
import logging
import numpy as np

from ditherer.dither_kernel import DitherKernel, DITHERS
from ditherer.ditherer import Ditherer
from .shapizer import Shapizer


OFFSET = math.sqrt(3)/2 # relationship of triangle width to height
PIXEL_HEIGHT = 10 # size of triangular pixels to draw in rectangular pixels


class Triangularizer(Shapizer):

    # FIXME: greylocation was formerly "data/gradient_tri_700_higher.npy"
    def __init__(self, default_outpath:str=None, greylocation:str=None):
        self.default_outpath = default_outpath
        self.greylocation = greylocation
        self.pixel_height = PIXEL_HEIGHT

    def _get_vertices_for_max_shape(self, rwidth:int, rheight:int, pixel_unitsize:bool = False)->list:
        '''Returns the maximal vertical triangle for the available rectangle
        '''

        # widest possible triangle width that will fit in the rectangle
        # this is either the rect width, or the rect height divided by offset,
        # as triangle width * offset == triangle height
        triwidth = min(rwidth, round(rheight / OFFSET))

        if (pixel_unitsize): # if we need to crop in terms of a unit size
            triwidth = math.floor(triwidth / self.pixel_height) * self.pixel_height

        trianglepoints = [(rwidth / 2 - triwidth / 2, triwidth * OFFSET), (rwidth / 2 + triwidth / 2, triwidth * OFFSET), (rwidth / 2, 0)]

        return trianglepoints
        
    def pixelate(self, im:Image.Image, width:int, height:int, outlocation:str=None, greyscale_list=None, greyoutlocation=None, rectangular=False, save_files=False)->Image.Image:
        '''
        For an entire image, make the pixels triangular, working off of a greyscale list (numpy array) if provided

        The assumption is that a greyscale_list is a dithered version of that list (everything at black or white) and so if one is provided, we save to the outdithered location.
        '''
        triwidth = (self.pixel_height) / OFFSET

        if greyoutlocation == None:
            greyoutlocation = self.greylocation

        greyset = [] # all the pixel shades of grey that we will dither

        im = self.crop_to_shape(im, self._get_vertices_for_max_shape(width, height, True), True)

        draw = ImageDraw.Draw(im)

        for row in range(math.floor(min(height, width * OFFSET) / self.pixel_height)):
            oddrow = (row % 2 == 1)
            greyset.append([])
            logging.debug(f"beginning row {row}")
            for col in range(0 - row, row + 1): 
                oddcol = (col % 2 == 1)
                xoffset = col * (triwidth / 2)
                if (oddrow and oddcol) or (not oddrow and not oddcol):
                    #upturned
                    triangle = [
                        (width / 2 - triwidth / 2 + xoffset, triwidth * OFFSET * (row + 1)), 
                        (width / 2 + triwidth / 2 + xoffset, triwidth * OFFSET * (row + 1)), 
                        (width / 2 + xoffset, triwidth * OFFSET * row)]
                else:
                    #downturned
                    triangle = [
                        (width / 2 - triwidth / 2 + xoffset, triwidth * OFFSET * row), 
                        (width / 2 + triwidth / 2 + xoffset, triwidth * OFFSET * row), 
                        (width / 2 + xoffset, triwidth * OFFSET * (row + 1))]

                # find average grey and print it
                if (greyscale_list is None): # if we are in the initial pixelating step, find avg color:
                    # make an image of just that triangle pixel so we can average the color
                    small_tri = self.crop_to_shape(im, triangle)

                    # crop away the image outside that pixel
                    small_tri = small_tri.crop(self._get_rect_for_polygon(triangle))

                    color = math.floor(self.find_avg_color(small_tri))

                else:
                    color = greyscale_list[row][col + row]

                draw.polygon(triangle, fill=(int(color),int(color),int(color),255))
                greyset[row].append(color)

        if save_files:
            super().save(outlocation, im, greyscale_list, greyset, greyoutlocation)

        return im
    
