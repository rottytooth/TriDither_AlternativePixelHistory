import sys, os
import datetime
import random
from PIL import Image, ImageOps
from cv2 import rectangle 
import numpy as np
import cv2
import uuid
from argparse import ArgumentParser
import textwrap
import logging
from ditherer.ditherer import Ditherer, DitherKernel
from shapizer import shapizer_factory
from tester.image_compare import ImageCompare
from tester.genetic_algorithm import GeneticAlgorithm
from tester.exhaustive_run import Exhaustive

parser = ArgumentParser()
parser.add_argument("process", nargs=1, help=textwrap.dedent("""
    dither: for an image or numpy array of greyscale values, dither from grey to b&w
"""))
parser.add_argument('-i', '--infile', type=str, help='input file')
parser.add_argument('-o', '--outfile', '--outpath', type=str, help='output file or folder')
parser.add_argument('-g', '--greylocation', type=str, help="location of npm data holding greyscale location; for 'score', 'dither', and 'ml', will read from this location. For 'pixelate', will write to it")
parser.add_argument('-k', '--kernel', type=str, help='dithering kernel')
parser.add_argument('-s', '--shape', type=str, default='square', help='pixel shape')
parser.add_argument('-d', '--downkernel', type=str, default='square', help='override the down-facing kernel')


def main():
    args = parser.parse_args()

    logFormatter = logging.Formatter("%(asctime)s  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    logging.info("Starting process: %s" % args.process)

    # set to default locations
    originalpath = "source_img/"
    pixelpath = "pixelated_img/"
    outpath = "out_img/"
    greypath = "data/"

    if args.process == ['pixelate']:
        # output both a pixelated image and an npm dataset of greyscale values
        pixelate(originalpath + args.infile, pixelpath + args.outfile, greypath + args.greylocation, args.shape)
    if args.process == ['dither']:
        # output a dithered image based on image or npm dataset and pixel shape
        dither(originalpath + args.infile, outpath + args.outfile, greypath + args.greylocation, args.shape, args.kernel)
    if args.process == ['score']:
        # score a single dithered image against its pixelated version (and npm dataset)
        score(pixelpath + args.infile, greypath + args.greylocation, args.shape)
        #FIXME: may want to pass outpath for this
    if args.process == ['ml']:
        # run ml model to find ideal kernel for pixel shape
        ml("pixelated_img/" + args.infile, greypath + args.greylocation, "out/" + args.outpath)
    if args.process == ['exhaustive']:
        exhaustive(pixelpath + args.infile, greypath + args.greylocation, outpath + args.outfile, "hexagon")

    logging.info("Process complete")


# Take an image and export a pixelated version, in the given pixel shape 
# Also exports a npm file for its greyscale pixel colors
def pixelate(original_image, outpath, greylocation, shape):
    shapizer = shapizer_factory.ShapizerFactory.GetShapizer(shape)
    shapizer.greylocation = greylocation

    # get size of image
    image = Image.open(original_image).convert('LA').convert("RGBA")
    width, height = image.size

    rect = (shape == 'righttriangle')

    shapizer.pixelate(image, width, height, outpath, rectangular=rect, save_files=True)

# Dither an image, based on the image's greyscale npm file and the
# selected dithering kernel. The shape selected must be the same as what
# the npm file was built from. Original image is used here only to
# determine the size of the output image (and can be overriden)
def dither(original_image, outpath, greylocation, shape, kernel):
    shapizer = shapizer_factory.ShapizerFactory.GetShapizer(shape)
    shapizer.greylocation = greylocation
    dither = DitherKernel(kernel)

    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    # call load_data with allow_pickle implicitly set to true
    greyset = np.load(shapizer.greylocation)
    # restore np.load for future normal usage
    np.load = np_load_old

    image = Image.open(original_image).convert('LA').convert("RGBA")
    width, height = image.size

    rowoffset = False
    if shape == "triangle":
        rowoffset = True

    ditherer = Ditherer(greyset)
    dithered_tri = ditherer.dither(kernel=dither, serpentine=False, rowoffset=rowoffset)

    if shape == "square":
        cv2.imwrite(outpath, dithered_tri)
    else:
        shapizer = shapizer_factory.ShapizerFactory.GetShapizer(shape)       

        rect = (shape == 'righttriangle')

        # here, pixelate just means draw the dithered image
        # this is indicated by including the grey data set (of b&w)
        shapizer.pixelate(image, width, height, outpath, dithered_tri, rectangular=rect, save_files=True) #, rectangular=True)

def score(infile, greyscale, outfile, shape, name):
    "Scores a single image"
    imc = ImageCompare(Ditherer(None))
    imc.greyscale = greyscale
    imc.original_img = infile

    # increase blurriness for large pixels
    if (shape != "square"):
        imc.sigma = 9.0

    # name="david_entry_test_"

    #NOTE: name is only used if saving the blurry files
    score = imc.scoreImageByLoc(infile, outfile, name)
    logging.info(f"score = {score.ssim}")
    return score.ssim


# def dither_and_score(infile, out_folder, greyfile, shape):
#     dither(infile, out_folder + "/test.png", greyfile, shape, kernel)

def make_dir_for_entry(path:str, dirname:str, iteration:int, idd:str)->None:
    idd = str(uuid.uuid4())
    if not os.path.exists(f"{path}/{dirname}"):
        try:
            os.makedirs(f"{path}/{dirname}")
        except:
            pass
    if not os.path.exists(f"{path}/{dirname}/{iteration}"):
        try:
            os.makedirs(f"{path}/{dirname}/{iteration}")
        except:
            pass # sometimes another thread will have created the folder after testing
    path = f"{path}/{dirname}/{iteration}/{idd}"
    os.mkdir(path)
    return path

def file_logging(run_indicator):
    logFormatter = logging.Formatter("%(asctime)s  %(message)s") # we already have this
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(f"logs/dithering_{run_indicator}.log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    rootLogger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

def exhaustive(infile, greyfile, outpath, shape):
    now = datetime.datetime.now()
    nowstr = f"{now.year}_{now.month}_{now.day}_{now.second}_{now.microsecond}"

    shapizer = shapizer_factory.ShapizerFactory.GetShapizer(shape)
    shapizer.greylocation = greyfile


    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    # call load_data with allow_pickle implicitly set to true
    greyset = np.load(shapizer.greylocation)
    # restore np.load for future normal usage
    np.load = np_load_old

    im = ImageCompare(Ditherer(greyset), infile, greyfile)
    im.sigma = 9.0

    start_ker = DitherKernel()
    start_ker.start_x = 1
    start_ker.start_y = 0
    start_ker.denom = 32
    start_ker.nums = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 32]]

    dt = Exhaustive(im, nowstr, 2, 5, 'hexagon', 1, 0, 32, start_ker)
    dt.run(shape)

def ml(infile, greyfile, outpath):
    now = datetime.datetime.now()
    nowstr = f"{now.year}_{now.month}_{now.day}_{now.second}_{now.microsecond}"

    file_logging(nowstr)

    iteration = 0
    idd = str(uuid.uuid4())
    newoutpath = make_dir_for_entry(outpath, nowstr, iteration, idd)
    # dither_and_score(infile, newoutpath, greyfile)

    im = ImageCompare()
    im.greyscale = greyfile
    im.original_img = infile
    dt = GeneticAlgorithm(im, nowstr)
    dt.run()


if __name__ == '__main__':
    main()
