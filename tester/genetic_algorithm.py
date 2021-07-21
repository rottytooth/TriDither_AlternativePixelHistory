import random
import math
import logging
import copy
import datetime
import numpy as np
import os
import multiprocessing

from ditherer.dither_kernel import DitherKernel
from ditherer.ditherer import Ditherer
from tester.image_compare import ImageCompare, ImageScore #, full_test_ditheralgo
from tester.test_utils import complete_random_dither

# defaults
TARGET_SCORE = 80.0 # FIXME: this depends on size of image and should not be set here

GENERATION_MAX = 5

POPULATION_SIZE = 6

class GeneticAlgorithm(object):

    def __init__(self, im:ImageCompare, now, generation_max=GENERATION_MAX, population_size=POPULATION_SIZE):
        self.im = im
        self.prev_attempts = []
        self.now = now
        self.generation_max = generation_max
        self.population_size = population_size

    def run(self, target_score:int=TARGET_SCORE)->DitherKernel:
        initial_score = 0.0
        generations = 0
        population = []

        #set up initial population
        for i in range(self.population_size):
            population.append(complete_random_dither())

        self._score_population(population)

        while initial_score < target_score and generations < self.generation_max:
            for i in range(self.population_size):
                pass

    @staticmethod
    def _get_score(member, return_dict, num, im):
        logging.info(f"Getting score for #{num}")
        return_dict[num] = im.imgtest(member, "genetic", False)
        logging.info(f"Completed work on #{num}")

    def _score_population(self, population):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []

        for i in range(0, len(population)):
            processes.append(multiprocessing.Process(target=GeneticAlgorithm._get_score, args=(population[i], return_dict, i, self.im)))
        for i in range(0, len(population)):
            processes[i].start()
        for i in range(0, len(population)):
            processes[i].join()

        print(return_dict)
