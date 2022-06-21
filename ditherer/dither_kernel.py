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

        
