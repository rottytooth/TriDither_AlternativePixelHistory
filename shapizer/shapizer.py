import sys
from enum import Enum, unique, auto
import numpy
import math
from PIL import Image, ImageDraw
import logging
import numpy as np

class Shapizer(object):

    def crop_to_shape(self, im:Image.Image, cropshape:list, invisible_to_black:bool=False)->Image.Image:
        '''Take a rectangular image and convert to the dimensions of the crop vertices provided (from get_vertices_for_max_shape)
        '''
        # convert to numpy (for convenience)
        imArray = numpy.asarray(im)

        # create mask
        maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)

        ImageDraw.Draw(maskIm).polygon(cropshape, outline=1, fill=1)

        mask = numpy.array(maskIm)

        # assemble new image (uint8: 0-255)
        newImArray = numpy.empty(imArray.shape,dtype='uint8')

        # colors (three first columns, RGB)
        newImArray[:,:,:3] = imArray[:,:,:3]

        # transparency (4th column)
        newImArray[:,:,3] = mask*255

        # back to Image from numpy
        newIm = Image.fromarray(newImArray, "RGBA")

        return newIm

    def _get_rect_for_polygon(self, shape):
        # for shape, find the containing rectangle
        first_x = sys.maxsize
        first_y = sys.maxsize
        last_x = 0
        last_y = 0
        for i in shape:
            if i[0] < first_x:
                first_x = i[0]
            if i[0] > last_x:
                last_x = i[0]
            if i[1] < first_y:
                first_y = i[1]
            if i[1] > last_y:
                last_y = i[1]
        return (first_x, first_y, last_x, last_y)

    
    def find_avg_color(self, im:Image.Image)->int:
        '''For a triangular shape, find all the (regular) pixels within it, 
        determine the average shade of grey
        '''
        width, height = im.size
        pixelcount = 0
        total = 0

        for x in range(width):
            for y in range(height):
                r,g,b,a = im.getpixel((x,y))
                if a > 0:
                    total += r # this assumes greyscale in rgb space; only looks at red value of pixel
                    pixelcount += 1
        if pixelcount == 0:
            return 0
        return round(total / pixelcount)

    # def get_greyscaledata(self, npyloc:str):
    #     "LOAD GREYSCALE DATA FROM 'greyset' AND DITHER"

    #     np_load_old = np.load

    #     # modify the default parameters of np.load
    #     np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    #     # call load_data with allow_pickle implicitly set to true
    #     greyset = np.load(npyloc)

    #     # restore np.load for future normal usage
    #     np.load = np_load_old

    #     # Inspect greyscale data
    #     # for i in range(0,len(greyset)):
    #     #     print(greyset[i])

    #     return greyset

    def save(self, outlocation, im, greyscale_list, greyset, greylocation):
        if outlocation is None:
            outlocation = self.default_outpath
        im.save(outlocation)
        logging.info(f"Saved to path {outlocation}")

        if greyscale_list is None:
            # if we've generated a greyset, it means this is the pixelated version for a non-pixelated original, so we will save it to greylocation 
            np.save(greylocation, greyset)
            logging.info(f"Saved greyset to {greylocation}")

