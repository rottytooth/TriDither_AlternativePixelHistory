import numpy
import math
from PIL import Image, ImageDraw
import logging
import numpy as np

from dither_kernel import DitherKernel, DITHERS
from ditherer import Ditherer


OFFSET = math.sqrt(3)/2 # relationship of triangle width to height
PIXEL_HEIGHT = 10 # size of triangular pixels to draw in rectangular pixels

SAVE_IMG_FILES = True


# following Globals should ONLY be used OUTSIDE of Triangularizer object
# and used to set parameters for Triangularizer

# original source image
INPUT_IMG = "source_img/david_tri_big.jpg"

# the pixelated, triangular version will go here
OUT_TRI_FINAL = "out_img/david.png"

# a numpy array with greyscale data for pixelated version
# this is used because pixelating the original image and averaging the color for each pixel is expensive. The Ditherer will use this to calculate dithered version of the image
GREY_LOCATION = "data/david.npy"


class Triangularizer(object):

    def __init__(self, outpath:str):
        self.outpath = outpath
#        self.outdithered = outdithered
        self.greylocation = GREY_LOCATION
        self.pixel_height = PIXEL_HEIGHT

    def get_vertices_for_max_shape(self, rwidth:int, rheight:int, pixel_unitsize:bool = False)->list:
        '''Returns the maximal vertical triangle for the available rectangle
        '''

        # widest possible triangle width that will fit in the rectangle
        # this is either the rect width, or the rect height divided by offset,
        # as triangle width * offset == triangle height
        triwidth = min(rwidth, round(rheight / OFFSET))

        if (pixel_unitsize): # if we need to crop in terms of a unit size
            triwidth = (triwidth % self.pixel_height) * self.pixel_height

        trianglepoints = [(rwidth / 2 - triwidth / 2, triwidth * OFFSET), (rwidth / 2 + triwidth / 2, triwidth * OFFSET), (rwidth / 2, 0)]

        return trianglepoints

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

        # if invisible_to_black:
        #     for a in range(0,len(newImArray)):
        #         for b in range(0,len(newImArray[a])):
        #             if newImArray[a][b][3] == 0:
        #                 newImArray[a][b] = [0, 0, 0, 0]

        # back to Image from numpy
        newIm = Image.fromarray(newImArray, "RGBA")

        return newIm

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
        
    def pixelate(self, im:Image.Image, width:int, height:int, outlocation:str=None, greyscale_list=None)->Image.Image:
        '''
        For an entire image, make the pixels triangular, working off of a greyscale list (numpy array) if provided

        The assumption is that a greyscale_list is a dithered version of that list (everything at black or white) and so if one is provided, we save to the outdithered location.
        '''
        triwidth = (self.pixel_height) / OFFSET

        greyset = [] # all the pixel shades of grey that we will dither

        im = self.crop_to_shape(im, self.get_vertices_for_max_shape(width, height, True), True)

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
                    small_tri = small_tri.crop((width / 2 - triwidth / 2 + xoffset, triwidth * OFFSET * row, width / 2 + triwidth / 2 + xoffset, triwidth * OFFSET * (row + 1)))

                    color = math.floor(self.find_avg_color(small_tri))

                else:
                    color = greyscale_list[row][col + row]

                draw.polygon(triangle, fill=(int(color),int(color),int(color),255))
                greyset[row].append(color)

        if SAVE_IMG_FILES:
            if outlocation is None:
                outlocation = self.outpath
            im.save(outlocation)
            logging.info(f"Saved to path {outlocation}")

            if greyscale_list is None:
                # if we've generated a greyset, it means this is the pixelated version for a non-pixelated original, so we will save it to greylocation 
                #FIXME: it might make sense for greylocation to be overrideable like outlocation
                np.save(self.greylocation, greyset)
                logging.info(f"Saved greyset to {self.greylocation}")


        return im
    
    def get_greyscaledata(self, npyloc:str=GREY_LOCATION):
        "LOAD GREYSCALE DATA FROM 'greyset' AND DITHER"

        np_load_old = np.load

        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        # call load_data with allow_pickle implicitly set to true
        greyset = np.load(npyloc)

        # restore np.load for future normal usage
        np.load = np_load_old

        # Inspect greyscale data
        # for i in range(0,len(greyset)):
        #     print(greyset[i])

        return greyset

if __name__ == '__main__':
    logFormatter = logging.Formatter("%(asctime)s  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    tri = Triangularizer(OUT_TRI_FINAL)

    # read image as RGB, convert to greyscale, and add alpha (transparency)
    im = Image.open(INPUT_IMG).convert('LA').convert("RGBA")

    iwidth, iheight = im.size

    # this creates the triangular-pixelled version
    iim = tri.pixelate(im, iwidth, iheight)



    # Take im data returned from Triangularizer and dither it:
    # ditherer = Ditherer(greyset)
    # dith_greys = ditherer.dither(DitherKernel("FloydSteinberg"), False)

    # im = im.getdata()
    # tri.pixelate(im, iwidth, iheight, dith_greys)

