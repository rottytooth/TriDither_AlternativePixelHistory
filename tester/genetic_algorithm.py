import random
import math
import logging
import copy
import datetime
import numpy as np
import os
import multiprocessing
import uuid
from functools import reduce
import numpy

from ditherer.dither_kernel import DitherKernel
from ditherer.ditherer import Ditherer
from tester.image_compare import ImageCompare, ImageScore #, full_test_ditheralgo

# defaults
TARGET_SCORE = 80.0 # FIXME: this depends on size of image and should not be set here

# how many generations before we stop
GENERATION_MAX = 500

# size of the first generation -- the next generations will be somewhat bigger
POPULATION_SIZE = 32

# how many top-scoring kernels are carried to the next generation
TOP_PICKS = 8

# how many new random kernels are introduced in the next generations
# NEW_KERNELS_PER_GEN = 16


class GeneticEntry(object):
    def __init__(self, loc:int, score:float, uuid:uuid, kernel:DitherKernel, gen:int):
        self.loc = loc
        self.score = score
        self.uuid = uuid
        self.kernel = kernel
        self.generation = gen

    def __str__(self):
        return f"loc: {self.loc}, score: {self.score}, uuid: {self.uuid}, kernel: ({self.kernel}), generation: {self.generation}"

class GeneticAlgorithm(object):

    def __init__(self, infilelist, greyfilelist, now, shape, outpath, generation_max=GENERATION_MAX, population_size=POPULATION_SIZE):
        self.prev_attempts = []
        self.now = now
        self.generation_max = generation_max
        self.population_size = population_size
        self.generations = 0
        self.shape = shape
        self.outpath = outpath

        self.imgcomplist = []
        for i in range(0, len(greyfilelist)):
            self.imgcomplist.append(ImageCompare(Ditherer(), infilelist[i], greyfilelist[i]))

    def run(self, target_score:int=TARGET_SCORE)->DitherKernel:
        initial_score = 0.0
        self.generations = 1
        population = []

        self.score = initial_score

        #set up initial population
        for i in range(self.population_size):
            population.append(DitherKernel.complete_random_dither())
        toplist = self.report_generation(population, [])

        while self.score < target_score and self.generations < GENERATION_MAX:
            self.generations += 1
            population = self.generate_variations(toplist)

            for i in range(3):
                population.append(DitherKernel.complete_random_dither())

            toplist = self.report_generation(population, toplist)

    def report_generation(self, population, toplist):

        logging.info("")
        logging.info(f"GENERATION {self.generations}")

        scores = self._score_population(population, self.generations)

        toplist.extend(scores)
        scores = toplist

        logging.info(f"Size of population: {len(scores)}")

        # re-sort with the toplist added
        scores = sorted(scores, key=lambda d: 0 - d.score)

        # remove doubles
        for i in range(len(scores)-1, 0, -1):
            if scores[i] != next((x for x in scores if x.kernel.nums == scores[i].kernel.nums), None):
                scores.remove(scores[i])
        # worst of population
        logging.info(f"WORST: {scores[len(scores)-1]}")

        top = scores[:int(TOP_PICKS)]

        # best of population
        for i in range(0, int(TOP_PICKS)):
            if len(top) > i:
                logging.info(f"{i+1} BEST: {top[i]}")

        return top

    def generate_variations(self, top):

        population = []
        for a in range(0, 8):
            for b in range(0, 3):
                for c in range(b + 1, 3):
                    population.append(self._crossover(top[b].kernel, top[c].kernel))

                population.append(self._mutate(top[b].kernel))

        for b in range(0, 1):
            for a in range(0, 8):
                if (random.randint(0,1) == 0):
                    population.append(self._crossover(top[a].kernel, DitherKernel.complete_random_dither()))
                else:
                    population.append(self._crossover(DitherKernel.complete_random_dither(), top[a].kernel))

        return population


    def _crossover(self, member1:DitherKernel, member2:DitherKernel)->DitherKernel:
        if (random.randint(0,1) == 0):
            retkernel = copy.deepcopy(member2)
            k2 = copy.deepcopy(member1)
        else:
            retkernel = copy.deepcopy(member1)
            k2 = copy.deepcopy(member2)

        end = len(retkernel.nums)
        if random.randint(0,1) == 0:
            end = len(k2.nums)

        for y in range(0, end): # rows
            if (y > 0 and y < len(k2.nums) and random.randint(0,1) == 0) or y >= len(retkernel.nums):
                numset = k2.nums[y]
                if len(retkernel.nums[0]) > len(numset):
                    numset += np.zeros(len(retkernel.nums[0]) - len(numset), dtype=int).tolist()
                elif len(retkernel.nums) < len(numset):
                    numset = numset[:len(retkernel.nums[0])]
                
                if y >= len(retkernel.nums):
                    retkernel.nums.append(numset)
                else:
                    retkernel.nums[y] = numset

        # if it can be reduced, reduce it
        try:
            allcells = numpy.array(retkernel.nums)
            allcells = numpy.append(allcells, retkernel.denom)
            flat = allcells.flatten()
            reduct = reduce(math.gcd, flat)

            retkernel.nums = [[int(x / reduct) for x in row] for row in retkernel.nums]
            retkernel.denom = sum(sum(retkernel.nums,[]))
        except Exception as e:
            logging.error(e)
            logging.error(f"bad kernel array: {retkernel.nums}")
            logging.error("SKIPPING REDUCE")

        try:
            # shave off any unused rows or cols
            testarray = np.array(retkernel.nums)
            colsums = testarray.sum(axis=0)
            for i in range(len(colsums) - 1, 0, -1):
                if colsums[i] == 0:
                    testarray = np.delete(testarray, i, 1)
                else:
                    break
            rowsums = testarray.sum(axis=1)
            for i in range(len(rowsums) - 1, 0, -1):
                if rowsums[i] == 0:
                    testarray = np.delete(testarray, i, 0)
                else:
                    break
        except Exception as e:
            logging.error(e)
            logging.error(f"bad kernel array: {retkernel.nums}")
        return retkernel

    def _mutate(self, member:DitherKernel)->DitherKernel:
        retDither = DitherKernel()
        retDither.start_x = member.start_x
        retDither.start_y = member.start_y
        retDither.denom = member.denom
        retDither.nums = copy.deepcopy(member.nums)

        # one to four random changes
        # FIXME: should not allow negatives
        # FIXME: should be in relation to denom (a 10000 denom, +1 will be useless)
        for a in range(0, random.randint(1, 12)):
            randx, randy = self.get_random_square(member)
            retDither.nums[randy][randx] += random.randint(1, 3)
            randx, randy = self.get_random_square(member, True)
            amount_sub = random.randint(1, 3)
            if (retDither.nums[randy][randx] > amount_sub):
                retDither.nums[randy][randx] -= amount_sub

#        retDither.reduce() # get smallest form of produced dither
        retDither.denom = sum(map(sum, retDither.nums))
        return retDither

    def __get_random_sq(self, state:DitherKernel):
        randy = random.randint(0, len(state.nums) - 1)
        if randy == state.start_y:
            randx = random.randint(state.start_x + 1, len(state.nums[randy]) - 1)
        else:
            randx = random.randint(0, len(state.nums[randy]) - 1)
        return randx, randy

    def get_random_square(self, state:DitherKernel, greaterThanZero:bool=False):
        """Determine a square to add to or subtract from"""
        randx, randy = self.__get_random_sq(state)

        if greaterThanZero:
            while state.nums[randy][randx] <= 0:
                randx, randy = self.__get_random_sq(state)

        return randx, randy


    @staticmethod
    def _get_score(member, return_dict, num:int, imgcomplist:list, now:str, gen:int, shape:str):
        logging.info(f"Getting score for #{num}")
        retscores = []

        idd = str(uuid.uuid4())

        for im in imgcomplist:
            retscores.append(im.imgtest(member, "genetic", False, now, gen, shape,idd=idd))
        idd = retscores[0][1] # This should be the same for all the entries
        scores = np.array(retscores)[:, 0].astype(float)
        return_dict[num] = (np.mean(scores), idd)
        logging.info(f"Completed work on #{num}")

    # returns population with best up front, worst in the back
    def _score_population(self, population, gen:int):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []

        for i in range(0, len(population)):
            processes.append(multiprocessing.Process(target=GeneticAlgorithm._get_score, args=(population[i], return_dict, i, self.imgcomplist, self.now, gen, self.shape)))
        for i in range(0, len(population)):
            processes[i].start()
        for i in range(0, len(population)):
            processes[i].join()

        # sort into the original order that processes started, not ended, to match the population list so the correct score is matched with the correct kernel
        retlist = []
        for a in range(0, len(return_dict)):
            retlist.append(
                GeneticEntry(
                    loc = a,
                    score = return_dict[a][0],
                    uuid = return_dict[a][1],
                    kernel = population[a],
                    gen = self.generations
                ))

        # sort by best score 
        retlist = sorted(retlist, key=lambda d: 0 - d.score)
        # logging.info(retlist)
        return retlist
