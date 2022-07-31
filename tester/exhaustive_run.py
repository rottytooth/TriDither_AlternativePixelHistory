import logging
import multiprocessing

from ditherer.dither_kernel import DitherKernel
from ditherer.ditherer import Ditherer
from tester.image_compare import ImageCompare, ImageScore #, full_test_ditheralgo
from tester.test_utils import complete_random_dither, arr_to_kernel

GEN_SIZE = 40

class Exhaustive(object):

    def __init__(self, im:ImageCompare, now:str, rows:int, cols:int, shape:str, start_x:int=1, start_y:int=0, denom:int=16, down_ker:DitherKernel=None):
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

    def _sum(self, arr): 
        sum=0
        for i in arr:
            sum = sum + i
        return(sum) 

    def produce_population_set(self, target:int, size:int, start:list, retlist:list):
        if (len(start) == size - 1):
            last = target - self._sum(start)
            if last > 0:
                start.append(last)
                retlist.append(start)
        else:
            for i in range(target):
                t = start[:]
                t.append(i)
                self.produce_population_set(target, size, t[:], retlist)

    def run(self, shape):
        logging.info(f"images writing to: out/{self.now}")

        retlist = []
        self.produce_population_set(self.denom,self.cols,[],retlist)
        # print(retlist)

        self._score_population(retlist)       

    @staticmethod
    def _get_score(member, return_dict, num, im:ImageCompare, denom, now, iteration, shape, rows, down_ker:DitherKernel=None):
        logging.info(f"Getting score for #{num}")
        return_dict[num] = im.imgtest(arr_to_kernel(member, denom, rows), "exhaustive", False, now, iteration, shape, down_ker=down_ker)
        logging.info(f"Completed work on #{num}")

    def _score_population(self, population):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        gencount = 0
        processes = []


        Exhaustive._get_score(population[0], return_dict, 0, self.im, self.denom, self.now, gencount, self.shape, self.rows, self.down_ker)

        # for j in range(0, len(population), GEN_SIZE):
        #     arrsize = range(j, min(j + GEN_SIZE, len(population)))
        #     gencount += 1
        #     logging.info(f"Beginning generation {gencount}")
        #     for i in arrsize:
        #         processes.append(multiprocessing.Process(target=Exhaustive._get_score, args=(population[i], return_dict, i, self.im, self.denom, self.now, gencount, self.shape, self.rows, self.down_ker)))
        #     for i in arrsize:
        #         processes[i].start()
        #     for i in arrsize:
        #         processes[i].join()

        print(return_dict)
        logging.info(return_dict)
