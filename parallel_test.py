import random
import math
import logging
import datetime
import os
import multiprocessing

from ditherer.ditherer import Ditherer
from tester.triangle_compare import TriangleCompare
from tester.simulated_annealing import DitherTester
from tester.test_utils import random_dither


def get_score(dt, neighbor, return_dict, num):
    logging.info(f"Getting score for neighbor #{num}")
    return_dict[num] = dt.get_score(neighbor)
    logging.info(f"Completed work on neighbor #{num}")

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
    dt = DitherTester(im, nowstr)

    rows = 2
    cols = 5
    denom = 16
    
    dt.im.greyscale = os.path.abspath("data/sanding_large_tri.npy")
    dt.im.original_img = os.path.abspath("out_img/sanding_large_tri.png")
    
    initial_state = random_dither(rows, cols, denom)

    neighbors = dt.get_neighbors(initial_state, 4)
    processes = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for i in range(0, len(neighbors)):
        processes.append(multiprocessing.Process(target=get_score, args=(dt, neighbors[i], return_dict, i)))
    for i in range(0, len(neighbors)):
        processes[i].start()

    for i in range(0, len(neighbors)):
        processes[i].join()

    logging.info("Done")
    print(return_dict)

    # final = dt.simulated_annealing(3, 8, 32, .7)
