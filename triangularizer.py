import numpy
import math
from PIL import Image, ImageDraw
import logging
import numpy as np

from dither_kernel import DitherKernel, DITHERS
from ditherer import Ditherer


OFFSET = math.sqrt(3)/2
PIXEL_HEIGHT = 10

SAVE_IMG_FILES = True

INPUT_IMG = "source_img/dtemkin_big.jpg"
# OUT_TRI_NOPIXEL = "out_img/out_tri2.png"
OUT_TRI_FINAL = "out_img/out_big.png"
OUT_TRI_DITHERED = "out_img/out_dith_big.png"

class Triangularizer(object):

    def __init__(self, inpath:str, outpath:str, outdithered:str):
        self.inpath = inpath
        self.outpath = outpath
        self.outdithered = outdithered

        self.pixel_height = PIXEL_HEIGHT

    def get_triangle(self, width, height): 
        '''Returns the maximal vertical triangle for the available rectangle
        '''
        trilength = min(width, round(height / OFFSET))
        return [(width / 2 - trilength / 2, height), (width / 2 + trilength / 2, height), (width / 2, 0)]

    def make_img_tri(self, im, cropshape, invisible_to_black=False):
        '''Take a rectangular image and convert to the dimensions of the crop shape provided
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

    def find_avg(self, im):
        '''For a triangular shape, find all the (regular) pixels within it, determine
        the average shade of grey
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
        return total / pixelcount
        
    def make_tri_pixels(self, im, width, height, greyscale_list=None):
        triheight = self.pixel_height

        greyset = [] # all the pixel shades of grey that we will dither

        # width, height = im.size
        trilength = min(height, width)

        im = self.make_img_tri(im, self.get_triangle(width, height), True)

        draw = ImageDraw.Draw(im)

        for row in range(math.floor(width / triheight)):
            oddrow = (row % 2 == 1) # odd rows
            greyset.append([])
            logging.debug(f"beginning row {row}")
            for col in range(0 - row, row + 1): 
                oddcol = (col % 2 == 1)
                xoffset = col * (triheight / 2)
                if (oddrow and oddcol) or (not oddrow and not oddcol):
                    #upturned
                    triangle = [
                        (width / 2 - triheight / 2 + xoffset, triheight * OFFSET * (row + 1)), 
                        (width / 2 + triheight / 2 + xoffset, triheight * OFFSET * (row + 1)), 
                        (width / 2 + xoffset, triheight * OFFSET * row)]
                else:
                    #downturned
                    triangle = [
                        (width / 2 - triheight / 2 + xoffset, triheight * OFFSET * row), 
                        (width / 2 + triheight / 2 + xoffset, triheight * OFFSET * row), 
                        (width / 2 + xoffset, triheight * OFFSET * (row + 1))]

                # find average grey and print it
                if (greyscale_list is None): # if we are in the initial pixelating step, find avg color:
                    # make an image of just that triangle pixel so we can average the color
                    small_tri = self.make_img_tri(im, triangle)

                    # crop away the image outside that pixel
                    small_tri = small_tri.crop((width / 2 - triheight / 2 + xoffset, triheight * OFFSET * row, width / 2 + triheight / 2 + xoffset, triheight * OFFSET * (row + 1)))

                    color = math.floor(self.find_avg(small_tri))
                else:
                    color = greyscale_list[row][col + row]

                draw.polygon(triangle, fill=(int(color),int(color),int(color),255))
                greyset[row].append(color)

        if SAVE_IMG_FILES:
            if greyscale_list is None:
                im.save(self.outpath)
                logging.info(f"Saved to path {self.outpath}")
            else:
                im.save(self.outdithered)
                logging.info(f"Saved to path {self.outdithered}")
        return greyset, im

if __name__ == '__main__':
    logFormatter = logging.Formatter("%(asctime)s  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    tri = Triangularizer(INPUT_IMG, OUT_TRI_FINAL, OUT_TRI_DITHERED)

    # read image as RGB, convert to greyscale, and add alpha (transparency)
    im = Image.open(INPUT_IMG).convert('LA').convert("RGBA")

    iwidth, iheight = im.size
    trilength = min(iheight, iwidth)

    # crop to proportions for eq triangle: (1 - OFFSET) * unit height, unit length width
    im = im.crop((0, trilength * (1 - OFFSET), trilength, trilength))
    iwidth, iheight = im.size # get current size

    # this creates the triangular-pixelled version
    greyset, iim = tri.make_tri_pixels(im, iwidth, iheight)

    np.save("data/greyset_bigme2", greyset)

    # save tri pixel colors to later dither
    # for i in range(0,len(greyset)):
    #     print(greyset[i])


    # LOAD GREYSCALE DATA FROM "greyset" AND DITHER

    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    # call load_data with allow_pickle implicitly set to true
    greyset = np.load("data/greyset_bigme2.npy")

    # restore np.load for future normal usage
    np.load = np_load_old


    ditherer = Ditherer(greyset)
    dith_greys = ditherer.dither(DitherKernel("FloydSteinberg"), False)

    im = im.getdata()
    tri.make_tri_pixels(im, iwidth, iheight, dith_greys)

    # WRITE OUT VALUES
    # with open('pixelgreys.py', 'w') as pixelgreys:
    #     i = 0
    #     for val in greyset:
    #         i += 1
    #         pixelgreys.write("pixelgreys.append(")
    #         for j in range(0, len(val)):
    #             if (j > 0):
    #                 pixelgreys.write(",")
    #             pixelgreys.write("[%s]" % (val[j]))
    #         pixelgreys.write(")\n")
