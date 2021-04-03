import random
import math
import logging
import copy
import datetime
import numpy as np

from dither_kernel import DitherKernel
from ditherer import Ditherer
from image_compare import ImageCompare, ImageScore

class DitherTester(object):

    def __init__(self):
        self.im = ImageCompare(Ditherer())
        self.prev_attempts = []

    def simulated_annealing(self, rows:int, cols:int, denom:int, final_score:int=None):
        """Peforms simulated annealing to find a solution"""
        initial_score = 0.0

        if final_score is None:
            final_score = 2.7
#        alpha = 0.01

        self.prev_attempts = []
        
        initial_state = self.im.random_dither(rows, cols, denom)
        current_score = initial_score

        # Start by initializing the current state with the initial state
        current_state = initial_state
        solution = current_state

        current_score = self.get_score(current_state)

        while current_score < final_score:
            neighbor = random.choice(self.get_neighbors(current_state))
            logging.info(f"Neighbor to test = {neighbor.nums}")

            # add to list of previous tries
            if not (np.array(current_state) in np.array(self.prev_attempts)):
                self.prev_attempts.append(current_state.nums)

            # Check if neighbor is best so far
            neighbor_score = self.get_score(neighbor)
            cost_diff = neighbor_score - current_score

            # if the new solution is better, accept it
            if cost_diff > 0:
                logging.info(f"Switching to neighbor: {neighbor.nums} with score of {neighbor_score}")
                solution = neighbor
                current_score = neighbor_score

            # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            else:
                if random.uniform(0, 1) < math.exp(-cost_diff / current_score) / 2.0:
                    logging.info(f"Switching RANDOMLY to neighbor: {neighbor.nums} with score {neighbor_score} from score of {current_score}")
                    solution = neighbor
                    current_score = neighbor_score

        return solution

    def get_score(self, state):
        "Fitness function: assumes rect here, child classes should override this test"
        scores = self.im.full_test_ditheralgo(state, "RANDTEST")
        score_tot = np.array(scores).sum()
        if (scores[-1] > 0.5): score_tot += 0.5 # extra points for last piece, since others cause "return early" behavior in image compare
        return score_tot


    def __get_random_sq(self,state):
        randy = random.randint(0, len(state.nums) - 1)
        if randy == state.start_y:
            randx = random.randint(state.start_x + 1, len(state.nums[randy]) - 1)
        else:
            randx = random.randint(0, len(state.nums[randy]) - 1)
        return randx, randy

    def get_random_square(self, state, greaterThanZero=False):
        """Determine a square to add to or subtract from"""
        randx, randy = self.__get_random_sq(state)

        if greaterThanZero:
            while state.nums[randy][randx] <= 0:
                randx, randy = self.__get_random_sq(state)

        return randx, randy

    def compareDithers(self, d1:DitherKernel, d2:DitherKernel) -> bool:
        d1num = np.array(d1.nums)
        d2num = np.array(d2.nums)

        if np.array_equal(d1num, d2num) and d1.start_x == d2.start_x and d1.start_y == d2.start_y and d1.denom == d2.denom:
            return True
        return False

    def get_neighbors(self, state:DitherKernel):
        """Build a set of neighbors with variations of your current kernel"""
        retset = []
        goal = random.randint(3, 6)
        count = 0
        DROPOUT = 100

        while len(retset) < goal:
            retDither = DitherKernel()
            retDither.start_x = state.start_x
            retDither.start_y = state.start_y
            retDither.denom = state.denom
            retDither.nums = copy.deepcopy(state.nums)

            for a in range(0, random.randint(1, 5)):
                randx, randy = self.get_random_square(state)
                retDither.nums[randy][randx] += 1
                randx, randy = self.get_random_square(state, True)
                retDither.nums[randy][randx] -= 1

            req_entries = False
            for b in range(0, len(retDither.nums)):
                if np.count_nonzero(retDither.nums[b]) == 0:
                    break
                req_entries = True

            if not np.array(retDither) in np.array(self.prev_attempts) and req_entries:
                retset.append(retDither)

            count += 1
            if (count > DROPOUT):
                raise("Could not produce neighbors")

        return retset

class DitherTesterTriangle(DitherTester):
    def __init__(self):
        super(DitherTesterTriangle, self).__init__()

    def get_score(self, state):
        return self.im.triangle_test(state, "test_tri", False)


if __name__ == '__main__':
    logFormatter = logging.Formatter("%(asctime)s  %(message)s")
    rootLogger = logging.getLogger()

    now = datetime.datetime.now()
    fileHandler = logging.FileHandler(f"logs/dithering_{now.year}_{now.month}_{now.day}_{now.second}_{now.microsecond}.log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    rootLogger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    dt = DitherTesterTriangle()
    final = dt.simulated_annealing(2, 4, 8, .7)

    logging.info("SOLUTION FOUND")
    logging.info(f"start_x = {final.start_x}")
    logging.info(f"start_y = {final.start_y}")
    logging.info(f"denom = {final.denom}")
    logging.info(f"nums = {final.nums}")
