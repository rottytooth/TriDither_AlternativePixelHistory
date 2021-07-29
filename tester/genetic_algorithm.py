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
from tester.test_utils import complete_random_dither

# defaults
TARGET_SCORE = 80.0 # FIXME: this depends on size of image and should not be set here

GENERATION_MAX = 10

POPULATION_SIZE = 4

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

    def __init__(self, im:ImageCompare, now, generation_max=GENERATION_MAX, population_size=POPULATION_SIZE):
        self.im = im
        self.prev_attempts = []
        self.now = now
        self.generation_max = generation_max
        self.population_size = population_size
        self.generations = 0

    def run(self, target_score:int=TARGET_SCORE)->DitherKernel:
        initial_score = 0.0
        self.generations = 1
        population = []

        self.score = initial_score

        #set up initial population
        for i in range(self.population_size):
            population.append(complete_random_dither())
        toplist = self.report_generation(population)

        while self.score < target_score and self.generations < GENERATION_MAX:
            self.generations += 1
            population = self.generate_variations(population, toplist)
            toplist = self.report_generation(population)

    def report_generation(self, population):

        logging.info(f"GENERATION {self.generations}")

        scores = self._score_population(population, self.generations)
        sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

        # worst of population
        logging.info("WORST")
        logging.info(sorted_scores[list(sorted_scores)[0]])
        logging.info(population[list(sorted_scores)[0]])
        logging.info("SECOND WORST")
        logging.info(sorted_scores[list(sorted_scores)[1]])
        logging.info(population[list(sorted_scores)[1]])

        # best of population
        top = []
        top.append(self._get_mut_res(sorted_scores.popitem(), population, 1))
        top.append(self._get_mut_res(sorted_scores.popitem(), population, 1))
        top.append(self._get_mut_res(sorted_scores.popitem(), population, 1))
        top.append(self._get_mut_res(sorted_scores.popitem(), population, 1))

        logging.info(f"BEST: {top[0]}")
        logging.info(f"SECOND BEST: {top[1]}")
        logging.info(f"THIRD BEST: {top[2]}")
        logging.info(f"FOURTH BEST: {top[3]}")

        return top

    def generate_variations(self, oldpop, top):

        population = []

        for a in range(0, 4):
            for b in range(0, 3):
                for c in range(b, 3):
                    population.append(self._crossover(top[b].kernel, top[c].kernel))

                population.append(self._mutate(top[b].kernel))

        return population

    def _get_mut_res(self, mut, population, gen):
        return GeneticEntry(
            loc=mut[0],
            score=mut[1][0],
            uuid=mut[1][1],
            kernel=population[mut[0]],
            gen=gen
        )

    def _crossover(self, member1:DitherKernel, member2:DitherKernel)->DitherKernel:
        k1 = member1
        k2 = member2
        if (random.randint(0,1) == 0):
            k1 = member2
            k2 = member1

        retdenom = k1.denom
        if (k1.denom != k2.denom):
            retdenom = k1.denom * k2.denom
        
        k1.nums = [[x * int(retdenom / k1.denom) for x in row] for row in k1.nums]
        k2.nums = [[x * int(retdenom / k2.denom) for x in row] for row in k2.nums]

        ybreak = random.randint(0, min(len(k1.nums), len(k2.nums)))
        xbreak = random.randint(0, min(len(k1.nums[0]), len(k2.nums[0])))
        while (xbreak <= k1.start_x and ybreak == k1.start_y):
            ybreak = random.randint(0, min(len(k1.nums), len(k2.nums)))
            xbreak = random.randint(0, min(len(k1.nums[0]), len(k2.nums[0])))

        retKernel = DitherKernel()
        retKernel.nums = []
        for y in range(0, len(k2.nums)):
            row = []
            for x in range(0, len(k2.nums[0])):
                if (x < xbreak or y < ybreak) and len(k1.nums[0]) > x+1 and len(k1.nums) > y+1:
                    row.append(k1.nums[y][x])
                else:
                    row.append(k2.nums[y][x])
            retKernel.nums.append(row)

        retKernel.start_x = k1.start_x
        retKernel.start_y = k1.start_y

        retKernel.denom = 0
        for y in range(0, len(retKernel.nums)):
            for x in range(0, len(retKernel.nums[y])):
                retKernel.denom += retKernel.nums[y][x]

        # if it can be reduced, reduce it
        allcells = numpy.array(retKernel.nums)
        allcells = numpy.append(allcells, retKernel.denom)
        flat = allcells.flatten()
        reduct = reduce(math.gcd, flat.tolist())

        retKernel.nums = [[int(x / reduct) for x in row] for row in retKernel.nums]
        retKernel.denom = int(retKernel.denom / reduct)

        return retKernel

    def _mutate(self, member:DitherKernel)->DitherKernel:
        retDither = DitherKernel()
        retDither.start_x = member.start_x
        retDither.start_y = member.start_y
        retDither.denom = member.denom
        retDither.nums = copy.deepcopy(member.nums)

        # one to four random changes
        for a in range(0, random.randint(1, 4)):
            randx, randy = self.get_random_square(member)
            retDither.nums[randy][randx] += 1
            randx, randy = self.get_random_square(member, True)
            retDither.nums[randy][randx] -= 1

#        retDither.reduce() # get smallest form of produced dither
        return retDither


    def __get_random_sq(self, state:DitherKernel)->[int,int]:
        randy = random.randint(0, len(state.nums) - 1)
        if randy == state.start_y:
            randx = random.randint(state.start_x + 1, len(state.nums[randy]) - 1)
        else:
            randx = random.randint(0, len(state.nums[randy]) - 1)
        return randx, randy

    def get_random_square(self, state:DitherKernel, greaterThanZero:bool=False)->[int,int]:
        """Determine a square to add to or subtract from"""
        randx, randy = self.__get_random_sq(state)

        if greaterThanZero:
            while state.nums[randy][randx] <= 0:
                randx, randy = self.__get_random_sq(state)

        return randx, randy


    @staticmethod
    def _get_score(member, return_dict, num:int, im:ImageCompare, now:str, gen:int):
        logging.info(f"Getting score for #{num}")
        return_dict[num] = im.imgtest(member, "genetic", False, now, gen)
        logging.info(f"Completed work on #{num}")

    def _score_population(self, population, gen:int):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []

        for i in range(0, len(population)):
            processes.append(multiprocessing.Process(target=GeneticAlgorithm._get_score, args=(population[i], return_dict, i, self.im, self.now, gen)))
        for i in range(0, len(population)):
            processes[i].start()
        for i in range(0, len(population)):
            processes[i].join()

        logging.info(return_dict)
        return return_dict
