import random
import math
import logging
import datetime
import os
import multiprocessing

from ditherer.ditherer import Ditherer, DitherKernel
from tester.triangle_compare import TriangleCompare
from tester.genetic_algorithm import GeneticAlgorithm
from tester.exhaustive_run import Exhaustive
from shapizer.righttriangularizer import Righttriangularizer


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

    # Triangle
    # im = TriangleCompare(Ditherer())
    # im.greyscale = os.path.abspath("data/Amanda_tri.npy")
    # im.original_img = os.path.abspath("out_img/Amanda_tri.png")

    #Right Triangle
    im = TriangleCompare(Ditherer(), Righttriangularizer)
    im.greyscale = os.path.abspath("data/david_rt_big.npy")
    im.original_img = os.path.abspath("out_img/david_rt_big.png")

#    dt = GeneticAlgorithm(im, nowstr)
    start_ker = DitherKernel()
    # [[0, 0, 11], [1, 3, 1]], /16 -- for down
    start_ker.start_x = 1
    start_ker.start_y = 0
    start_ker.denom = 16
    start_ker.nums = [[0, 0, 11], [1, 3, 1]]
    dt = Exhaustive(im, nowstr, 2, 3, 1, 0, 16, start_ker)
    # dt.test_problem()
    dt.run()
