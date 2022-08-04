import logging
import random
import numpy as np

from ditherer.dither_kernel import DitherKernel


def complete_random_dither():
    dithersize = random_dither_size()
    return random_dither(dithersize['rows'], dithersize['cols'], dithersize['denom'])

def random_dither_size():
    # FIXME: there should be coherence between numer of cells and denom
    return {
        'rows' : random.randint(2,4),
        'cols' : random.randint(3,8),
        'denom' : random.randint(4,48)
    }

def arr_to_kernel(nums:list, denom:int, rows:int, cols:int, start_x:int)->DitherKernel:
    start_y = 0 #FIXME: will support for start_y be needed?
    retDith = DitherKernel()
    retDith.start_x = start_x
    retDith.start_y = start_y
    nums = np.zeros(start_x + 1).astype('int64').tolist() + nums
    retDith.nums = [nums[x:x+cols] for x in range(0, len(nums),cols)]
    retDith.denom = denom
    return retDith 

def random_dither(matrixrows:int=0, matrixcols:int=0, denom:int=0) -> DitherKernel:
    """Generate a random random dither kernel within specified parameters"""
    # for matrix we are returning
    width = matrixcols if matrixcols > 0 else random.randint(3,8)
    height = matrixrows if matrixrows > 0 else random.randint(2,3)

    retDither = DitherKernel()
    retDither.start_x = random.randint(1, max(1, width - 2)) # start x at least one before the end
    retDither.start_y = 0 # always starts somewhere in the first row
    retDither.nums = []
    retDither.denom = denom if denom > 0 else random.randint(width, 50) # denominator of error diffusion

    denom_count = retDither.denom # how much is left before we have 100% distribution of error

    # distribute the total num value by row
    row_distribution = [0]

    rowtries = 0
    while 0 in row_distribution and rowtries < 8:
        row_distribution = np.round(np.random.dirichlet(np.ones(matrixrows),size=1)*retDither.denom, 0)[0].astype(int).tolist()
        rowtries += 1

    for y in range(0, height):
        if y < retDither.start_y:
            # append row with all 0s
            row = [0] * width
        elif y == retDither.start_y:
            # distribute but leave first few cells for 0s
            row = [0] * (retDither.start_x + 1)
            row += np.round(np.random.dirichlet(np.ones(width - retDither.start_x - 1),size=1)*row_distribution[y], 0)[0].astype(int).tolist() # array in array
        else:
            # totally random distribution for this row
            row = np.round(np.random.dirichlet(np.ones(width),size=1)*row_distribution[y], 0)[0].astype(int).tolist()

        retDither.nums.append(row)
    
    logging.info(f"start_x: {retDither.start_x}")
    logging.info(f"start_y: {retDither.start_y}")
    logging.info(f"denom: {retDither.denom}")
    logging.info(f"nums: {retDither.nums}")
    
    return retDither