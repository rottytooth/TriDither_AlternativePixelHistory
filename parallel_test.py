import random
import math
import logging
import datetime
import os
import multiprocessing
from ml_model import DitherTesterTriangle

def get_score(dt, neighbor, return_dict, num):
    logging.info(f"Getting score for neighbor #{i}")
    return_dict[num] = dt.get_score(neighbor)
    logging.info(f"Completed work on neighbor #{i}")

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
    dt = DitherTesterTriangle(nowstr)

    rows = 2
    cols = 5
    denom = 16
    
    dt.im.greyscale = os.path.abspath("data/david_rt_big.npy")
    dt.im.original_img = os.path.abspath("source_img/david_tri_big.jpg")
    
    initial_state = dt.im.random_dither(rows, cols, denom)

    neighbors = dt.get_neighbors(initial_state, 160)
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
