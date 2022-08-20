import logging
import multiprocessing
import numpy as np
from PIL import Image
import time

from ditherer.dither_kernel import DitherKernel
from tester.image_compare import ImageCompare, ImageScore #, full_test_ditheralgo
from tester.test_utils import complete_random_dither, arr_to_kernel

GEN_SIZE = 20
ROW_MIN = 2

# run one and stop
TEST_RUN = False

# don't score
SKIP_SCORING = False

# start at a later generation
GEN_START = 1

class Exhaustive(object):

    def __init__(self, im:ImageCompare, now:str, rows:int, cols:int, shape:str, start_x:int=1, start_y:int=0, denom:int=16, serpentine:bool=False, down_ker:DitherKernel=None, row_min:int=ROW_MIN):
        self.im = im
        self.now = now
        self.denom = denom
        self.rows = rows
        self.cols = cols
        self.start_x = start_x
        self.start_y = start_y
        self.population = []
        self.down_ker = down_ker
        self.shape = shape
        self.serpentine = serpentine

    def _sum(self, arr): 
        sum=0
        for i in arr:
            sum = sum + i
        return(sum) 

    def load_kernel_list(self, kernel): 
        if len(kernel) == self.kernel_length - 1:
            remaining_value = self.denom - np.sum(kernel)
            if remaining_value < 0:
                raise Exception("impossible kernel rejected")
            kernel = np.append(kernel, remaining_value)

            test_arr = np.zeros(self.start_x + 1) # FIXME: this assumes start_y of zero
            test_arr = np.append(test_arr, kernel)

            # make sure that each row has a value > 0 or don't add
            test_arr = np.array_split(test_arr, self.rows)
            sums = np.sum(test_arr, axis=1)
            if np.where(sums < ROW_MIN)[0].size == 0: # if each rows sums to the minimum
                self.kernel_list.append(kernel.tolist())
            self.initial_kernel_loaded = True
            return
        elif not self.initial_kernel_loaded:
            nkernel = np.append(kernel, self.initial_kernel[len(kernel)])
            self.load_kernel_list(nkernel)
        start_count = 0 #self.initial_kernel[len(kernel)] + 1
        end_count = self.denom - np.sum(kernel)
        for i in range(start_count, end_count):
            self.load_kernel_list(np.append(kernel, i))

    def run(self):
        logging.info(f"images writing to: out/{self.now}")

        # set initial state
        self.kernel_list = []

        # assumes start_y of zero for now and no back-propogation
        self.kernel_length = (self.rows - 1) * self.cols + (self.cols - self.start_x - 1)
        self.initial_kernel = np.zeros(self.kernel_length).astype('int64')

        self.initial_kernel[self.cols - self.start_x - 2] = 1 # end of first row
        
        # for a three- or more-rowed kernel, this should fill in 1s at the end of the row
        for i in range(1, self.rows - 1): 
            self.initial_kernel[self.cols - self.start_x + i * self.cols] = 1

        # whatever is left over is stuck at the end
        self.initial_kernel[-1] = self.denom - np.sum(self.initial_kernel) 

        # this should be a check to make sure we've generated the correct starting state: kernel should have each row ending with 1 except the last row, which is the denom minus those 1s.
        self.denom = np.sum(self.initial_kernel) # denominator
        self.initial_kernel_loaded = False
        self.load_kernel_list(np.array([]).astype('int64'))

        self._score_population()

 
    @staticmethod
    def _get_score(member, return_dict, num, im:ImageCompare, denom, now, iteration, shape, rows, cols, start_x, serpentine, down_ker:DitherKernel=None):
        
        #FIXME: Why do we need to do all this again? Without this, nothing is logged
        logFormatter = logging.Formatter("%(asctime)s  %(message)s")
        rootLogger = logging.getLogger()
        rootLogger.setLevel(logging.INFO)
        fileHandler = logging.FileHandler(f'../logs/exh_{now}.log')
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        logging.info(f"Getting score for #{num}")
        return_dict[num] = im.imgtest(arr_to_kernel(member, denom, rows, cols, start_x), "exhaustive", serpentine, now, iteration, shape, down_ker=down_ker, skip_scoring=SKIP_SCORING)
        logging.info(f"Completed work on #{num}")

    def _score_population(self):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        gencount = GEN_START - 1
        processes = []

        if (TEST_RUN):
            Exhaustive._get_score(self.kernel_list[0], return_dict, 0, self.im, self.denom, self.now, gencount, self.shape, self.rows, self.cols, self.start_x, self.serpentine, self.down_ker)
            return

        starting_kernel = (GEN_START - 1) * GEN_SIZE

        for j in range(starting_kernel, len(self.kernel_list), GEN_SIZE):
            arrsize = range(j, min(j + GEN_SIZE, len(self.kernel_list) - j))
            gencount += 1
            time.sleep(1) # to give processes chance to close
            if gencount % 25 == 0:
                time.sleep(19)
            logging.info(f"Beginning generation {gencount}")
            for i in arrsize:
                processes.append(multiprocessing.Process(target=Exhaustive._get_score, args=(self.kernel_list[i], return_dict, i, self.im, self.denom, self.now, gencount, self.shape, self.rows, self.cols, self.start_x, self.serpentine, self.down_ker)))
            for i in arrsize:
                processes[i - starting_kernel].start()
            for i in arrsize:
                processes[i - starting_kernel].join()

        print(return_dict)
        logging.info(return_dict)
