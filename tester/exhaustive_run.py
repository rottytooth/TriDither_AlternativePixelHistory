import logging
import math
from multiprocessing import Process, Manager
import numpy as np
# import psutil
import time

from ditherer.ditherer import Ditherer, DitherKernel
from tester.image_compare import ImageCompare, ImageScore #, full_test_ditheralgo
from tester.test_utils import complete_random_dither, arr_to_kernel

GEN_SIZE = 48
ROW_MIN = 2

BLUR_LEVEL = 18.0 #FIXME: This should be calculated based on pixel size and maybe image size

# run one and stop
TEST_RUN = False

# don't score
SKIP_SCORING = False

# start at a later generation
GEN_START = 1

class Exhaustive(object):

    def __init__(self, infile:str, greyfile:str, now:str, rows:int, cols:int, shape:str, start_x:int=1, start_y:int=0, denom:int=16, serpentine:bool=False, down_ker:DitherKernel=None, row_min:int=ROW_MIN):
        self.infile = infile
        self.greyfile = greyfile
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
    def _get_score(member, return_dict, num, infile, greyfile, denom, now, iteration, shape, rows, cols, start_x, serpentine, down_ker:DitherKernel=None):

        #FIXME: Why do we need to do all this again? Without this, nothing is logged
        logFormatter = logging.Formatter("%(asctime)s  %(message)s")
        rootLogger = logging.getLogger()
        rootLogger.setLevel(logging.INFO)
        fileHandler = logging.FileHandler(f'../logs/exh_{now}.log')
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
        
        logging.info(f"Getting score for #{num}")
        np_load_old = np.load
        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        # call load_data with allow_pickle implicitly set to true
        greyset = np.load(greyfile)
        # restore np.load for future normal usage
        np.load = np_load_old

        im = ImageCompare(Ditherer(greyset), infile, greyfile)
        im.sigma = BLUR_LEVEL
        del greyset
        return_dict[num] = im.imgtest(arr_to_kernel(member, denom, rows, cols, start_x), "exhaustive", serpentine, now, iteration, shape, down_ker=down_ker, skip_scoring=SKIP_SCORING)
        logging.info(f"Completed work on #{num}")

    def _score_population(self):
        manager = Manager()
        return_dict = manager.dict()
        gencount = GEN_START - 1

        if (TEST_RUN):
            Exhaustive._get_score(self.kernel_list[0], return_dict, 0, self.infile, self.greyfile, self.denom, self.now, gencount, self.shape, self.rows, self.cols, self.start_x, self.serpentine, self.down_ker)
            return

        try:

            starting_kernel = (GEN_START - 1) * GEN_SIZE

            for j in range(starting_kernel, len(self.kernel_list), GEN_SIZE):
                fails = 0

                # for proc in psutil.process_iter():
                #     logging.warn(f"Open Files: {proc.open_files()}")
                processes = []


                arrsize = range(0, min(GEN_SIZE, len(self.kernel_list) - j))
                gencount += 1

                logging.info(f"Beginning generation {gencount} / {math.ceil(len(self.kernel_list) / GEN_SIZE) + 1}")
                for i in arrsize:
                    processes.append(Process(target=Exhaustive._get_score, args=(self.kernel_list[i], return_dict, i, self.infile, self.greyfile, self.denom, self.now, gencount, self.shape, self.rows, self.cols, self.start_x, self.serpentine, self.down_ker)))
                for i in arrsize:
                    try:
                        processes[i].start()
                    except Exception as e:
                        logging.error(f"ERROR STARTING PROCESS: {e.__class__.__name__}, {e}")
                        logging.error("PAUSING")
                        fails += 1
                        if fails > 3:
                            break
                        time.sleep(3)
                        logging.error("TRYING AGAIN")
                        try:
                            processes[i].start()
                        except Exception:
                            logging.error("FAILED AGAIN, SKIPPING...")
                        # raise e
                for i in arrsize:
                    try:
                        processes[i].join()
                    except Exception as e:
                        logging.error(f"ERROR JOINING PROCESS: {e.__class__.__name__}, {e}")
                        raise e # or catch here
        finally:
            logging.info(return_dict)
