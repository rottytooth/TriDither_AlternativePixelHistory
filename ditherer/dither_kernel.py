import logging
import random
import numpy as np

DITHERS = {
    "FloydSteinberg" :
    { 
        "start_x" : 1,
        "start_y" : 0,
        "denom" : 16,
        "nums" : [[0, 0, 7], [3, 5, 1]]
    },
    "ZFan" :
    { 
        "start_x" : 2,
        "start_y" : 0,
        "denom" : 48,
        "nums" : [[0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]]
    },
    "ShiauFan" :
    { 
        "start_x" : 2,
        "start_y" : 0,
        "denom": 8,
        "nums": [[0, 0, 0, 4], [1, 1, 2, 0]]
    },
    "ShiauFan2" :
    { 
        "start_x" : 3,
        "start_y" : 0,
        "denom" : 16,
        "nums" : [[0, 0, 0, 0, 8], [1, 1, 2, 4, 0]]
    },
    "Jarvis, Judice and Ninke" :
        { "start_x" : 2,
        "start_y" : 0,
        "denom" : 48,
        "nums" : [[0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]]
    },
    "Stucki" : 
        { "start_x" : 2,
        "start_y" : 0,
        "denom" : 42,
        "nums" : [[0, 0, 0, 8, 4], [2, 4, 8, 4, 2], [1, 2, 4, 2, 1]]
    },
    "Burkes" :
        { "start_x" : 2,
        "start_y" : 0,
        "denom" : 16,
        "nums" : [[0, 0, 0, 4, 2], [1, 2, 4, 2, 1]]
    },
    "Sierra" :
        { "start_x" : 2,
        "start_y" : 0,
        "denom" : 32,
        "nums" : [[0, 0, 0, 5, 3], [2, 4, 5, 4, 2], [0, 2, 3, 2, 0]]
    },
    "Two-Row Sierra" :
        { "start_x" : 2,
        "start_y" : 0,
        "denom" : 16,
        "nums" : [[0, 0, 0, 4, 3], [1, 2, 3, 2, 1]]
    },
    "FilterLite" :
        { "start_x" : 1,
        "start_y" : 0,
        "denom": 4,
        "nums" : [[0, 0, 2], [1, 1, 0]]
    },
    "Atkinson" :
        { "start_x" : 1,
        "start_y" : 0,
        "denom" : 8,
        "nums" : [[0, 0, 1, 1], [1, 1, 1, 0], [0, 1, 0, 0]]
    },

    # TEST DITHERS

    "Temkin1":
        { "start_x" : 2,
        "start_y" : 0,
        "denom" : 8,
        "nums" : [[0, 0, 0, 3, 1], [1, 0, 1, 2]]
    },

    "FloydSteinbergTriVariation":
        { "start_x" : 2,
        "start_y" : 0,
        "denom" : 13,
        "nums" : [[0, 0, 0, 4, 2], [1, 0, 0, 1, 2, 1, 2]]
    },
    "FSTriTwo":
    {
        "start_x" : 1,
        "start_y" : 0,
        "denom" : 18,
        "nums" : [[0, 0, 4, 2, 1], [2, 4, 1, 1, 2, 1]]
    },
    "StephensonArceForTriangles":
    {
        "start_x" : 3,
        "start_y" : 0,
        "denom" : 200,
        "nums" : [[0,0,0,0,0,32], [12,12,26,26,30,12,16], [5,0,12,0,12,0,5]]
    },
    "StephensonArceForTrianglesAlt":
    {
        "start_x" : 3,
        "start_y" : 0,
        "denom" : 200,
        "nums" : [[0,0,0,0,32], [12,12,26,26,30,12,16], [5,6,6,6,6,2,3]]
    },
    "StephensonArce":
    {
        "start_x" : 1,
        "start_y" : 0,
        "denom" : 200,
        "nums" : [[0,0,32],[12,26,30,16],[12,26,12,0],[5,12,12,5]]
    }
}

class DitherKernel(object):
    def __init__(self, name:str=None):
        if (name is not None):
            self.start_x = int(DITHERS[name]['start_x'])
            self.start_y = int(DITHERS[name]['start_y'])
            self.nums = list(DITHERS[name]['nums'])
            self.denom = int(DITHERS[name]["denom"])

    def __str__(self):
        return f"nums: {self.nums}, denom: {self.denom}, start: [{self.start_x}, {self.start_y}]"

    def clone(self):
        dk = DitherKernel()
        dk.start_x = self.start_x
        dk.start_y = self.start_y
        dk.nums = self.nums[:]
        dk.denom = self.denom
        return dk

    def reduce(self):
        """Reduce a kernel to its minimal size: removing unused columns (columns that only have zeros), unused rows
        """
        #FIXME: this can only do the edges, the left or rightmost, or the bottom
        allZero = True
        toDelete = []
        for r in self.nums:
            for c in self.nums[r]:
                if self.nums[r][c] != 0:
                    allZero = False
                    break
            if allZero:
                toDelete.append(r)
        
        for r in range(toDelete-1, 0, -1):
            del self.nums[r]

        allZero = True
        toDelete = []
        for c in self.nums[0]:
            for r in self.nums[c]:
                if self.nums[r][c] != 0:
                    allZero = False
                    break
            if allZero:
                toDelete.append(c)
        
        for c in range(toDelete-1, 0, -1):
            for r in len(self.nums):
                del self.nums[r][c]

    @staticmethod
    def arr_to_kernel(nums:list, denom:int, rows:int, cols:int, start_x:int):
        start_y = 0 #FIXME: will support for start_y be needed?
        retDith = DitherKernel()
        retDith.start_x = start_x
        retDith.start_y = start_y
        nums = np.zeros(start_x + 1).astype('int64').tolist() + nums
        retDith.nums = [nums[x:x+cols] for x in range(0, len(nums),cols)]
        retDith.denom = denom
        return retDith         

    #FIXME: Does not currently respect denom at all. This seems to only be used by Simulated Annealing
    @staticmethod
    def random_dither(matrixrows:int=0, matrixcols:int=0, denom:int=0):
        """Generate a random random dither kernel within specified parameters"""
        # for matrix we are returning
        width = matrixcols if matrixcols > 0 else random.randint(3,8)
        height = matrixrows if matrixrows > 0 else random.randint(2,3)

        retDither = DitherKernel()
        retDither.start_x = random.randint(1, max(1, width - 2)) # start x at least one before the end
        retDither.start_y = 0 # always starts somewhere in the first row
        retDither.nums = []
        retDither.denom = denom if denom > 0 else random.randint(width, 50) # denominator of error diffusion

        # denom_count = retDither.denom # how much is left before we have 100% distribution of error

        # distribute the total num value by row
        row_distribution = [0]

        rowtries = 0
        while 0 in row_distribution and rowtries < 8:
            row_distribution = np.round(np.random.dirichlet(np.ones(matrixrows),size=1)*retDither.denom, 0)[0].astype(int).tolist()
            rowtries += 1

        for y in range(0, retDither.start_y):
            retDither.nums.append([0] * width)
        for y in range(retDither.start_y, height):
            if y == retDither.start_y:
                row = [0] * (retDither.start_x + 1)
            else:
                row = []
            for x in range (len(row), width):
                row.append(random.randint(0, 3) + random.randint(0, 3) + random.randint(0, 3))
            retDither.nums.append(row)

        retDither.denom = sum(sum(retDither.nums,[]))
        
        logging.info(f"NEW KERNEL: {retDither}")
        
        return retDither

    @staticmethod
    def complete_random_dither():
        dithersize = DitherKernel.random_dither_size()
        return DitherKernel.random_dither(dithersize['rows'], dithersize['cols'])

    @staticmethod
    def random_dither_size():
        # FIXME: there should be coherence between numer of cells and denom
        return {
            'rows' : random.randint(2,4),
            'cols' : random.randint(3,8)
        }
