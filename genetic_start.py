import random
import math
import logging
import datetime
import os
import multiprocessing

from ditherer.ditherer import Ditherer
from tester.triangle_compare import TriangleCompare
from tester.genetic_algorithm import GeneticAlgorithm



if __name__ == '__main__':
    logFormatter = logging.Formatter("%(asctime)s  %(message)s")
    rootLogger = logging.getLogger()

    now = datetime.datetime.now()
    nowstr = f"{now.year}_{now.month}_{now.day}_{now.second}_{now.microsecond}"
    fileHandler = logging.FileHandler(f"logs/dithering_{nowstr}.log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    rootLogger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    im = TriangleCompare(Ditherer())
    im.greyscale = os.path.abspath("data/sanding_large_tri.npy")
    im.original_img = os.path.abspath("out_img/sanding_large_tri.png")
    
    dt = GeneticAlgorithm(im, nowstr)
    dt.run()
