import sys
import os
import logging
import uuid
import math
import random
import datetime
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import filters, data, exposure
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps 

from ditherer.dither_kernel import DITHERS, DitherKernel
from ditherer.ditherer import Ditherer

PATH = 'out/'


class ImageScore(object):
    def __init__(self, blur_level:np.float64, ssim:np.float64):
        self.blur_level = blur_level
        # self.mse = mse # no longer used
        self.ssim = ssim
        self.ssim_c = 0.0 # with contrast


class ImageCompare(object):
    """
    Comparer of dithers, returns a score on visual likeness. To be overriden for different dither patterns
    """

    def __init__(self, d:Ditherer, original_img:str=None, greyscale:str=None):
        self.sigma = 1.5
        self.d = d

        # path to the original image: will be used for sizing information
        self.original_img = original_img

        # path to a numpy dataset holding greyscale info for pixelated version
        self.greyscale = greyscale

    # NOTE: Not currently used
    def mse(self, imageA:np.ndarray, imageB:np.ndarray) -> np.float64:
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        
        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

    def scoreImage(self, original:np.ndarray, compare:np.ndarray, name:str) -> ImageScore:

        # cv2.imwrite(f"{PATH}{name}_original_nb.PNG", original)
        # cv2.imwrite(f"{PATH}{name}_compare_nb.PNG", compare)

        blurred_original = filters.gaussian(original, sigma=self.sigma, preserve_range=True)
        blurred_compare = filters.gaussian(compare, sigma=self.sigma, preserve_range=True)

        # uncomment to write blurred images to files
        cv2.imwrite(f"{PATH}{name}_original.PNG", blurred_original)
        cv2.imwrite(f"{PATH}{name}_compare.PNG", blurred_compare)

        score = ImageScore(self.sigma, ssim(blurred_original, blurred_compare))

        # NOTE: the following increases intensity and then re-blurs to test
        # The idea is to test well for Atkinson and other "mid-range" dithers
        # However, rescale_intensity won't work bc most images are already well-exposed here
        # so it ends up not changing it. We need a method that increases contrast regardless
        # of existing contrast, and find the right amount to do it

        # original_c = exposure.rescale_intensity(original)
        # compare_c = exposure.rescale_intensity(compare)

#        blurred_c_original = filters.gaussian(original_c, sigma=self.sigma, preserve_range=True)
#        blurred_c_compare = filters.gaussian(compare_c, sigma=self.sigma, preserve_range=True)

        # uncomment to write blurred images to files
#        cv2.imwrite(f"{PATH}{name}_original_cont.PNG", blurred_c_original)
#        cv2.imwrite(f"{PATH}{name}_compare_cont.PNG", blurred_c_compare)

        # test with reset contrast
#        score.ssim_c = ssim(blurred_c_original, blurred_compare)

        return score

    def random_img(self, original:np.ndarray) -> np.ndarray:
        """Generate a completely random image as baseline test"""
        x = original.copy()
        for j in range(0, len(x)):
            for i in range(0, len(x[j])):
                x[j][i] = random.randint(0, 1) * 255

        # if filename != "":
        #     cv2.imwrite(PATH + filename, x)
        return x

    def random_dither(self, matrixrows:int=0, matrixcols:int=0, denom:int=0) -> DitherKernel:
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
        row_distribution = np.round(np.random.dirichlet(np.ones(matrixrows),size=1)*retDither.denom, 0)[0].astype(int).tolist()

        for y in range(0, height):
            if y < retDither.start_y:
                # append row with all 0s
                row = [0] * width
            elif y == retDither.start_y:
                # distribute but leave first few cells for 0s
                row = [0] * (retDither.start_x + 1)
                row += np.round(np.random.dirichlet(np.ones(width - retDither.start_x),size=1)*row_distribution[y], 0)[0].astype(int).tolist() # array in array
            else:
                # totally random distribution for this row
                row = np.round(np.random.dirichlet(np.ones(width),size=1)*row_distribution[y], 0)[0].astype(int).tolist()

            retDither.nums.append(row)
        
        logging.info(f"start_x: {retDither.start_x}")
        logging.info(f"start_y: {retDither.start_y}")
        logging.info(f"denom: {retDither.denom}")
        logging.info(f"nums: {retDither.nums}")
        return retDither

    def test_itself(self, image:np.ndarray, filename:str):
        "Test an image against itself, as control"
        self.run_test(image, image, filename, "itself", False)

    def test_cutoffs(self, image:np.ndarray, filename:str):
        'Round images to black or white with no dithering and compare'
        im_pil = Image.fromarray(image)
        im2 = ImageOps.posterize(im_pil, 1)
        # im2.show()
        self.run_test(np.array(im2), image, filename, "Cutoffs", False)

    def test_random(self, image:np.ndarray, filename:str):
        'Test against randomized pixels'
        rnd = self.random_img(image)
        self.run_test(rnd, image, filename, "Random", False)

        
    def run_test(self, fs:np.ndarray, image:np.ndarray, filename:str, dithername:str, serpentine:bool)->np.float64:
        "Just calls scoreImage, and logs result, returns score"
        score1 = self.scoreImage(image, fs, filename + dithername + ('serpentine' if serpentine else 'ltor'))
        logging.info(f"{dithername} for {filename}, {'serpentine' if serpentine else 'left to right'}:")
        logging.info(f"\tssim: {score1.ssim}")
        # print(f"\tssim_c: {score1.ssim_c}")
        return score1.ssim


    # NOTE: this needs some re-thinking in terms of source images -- more variety
    def full_test_ditheralgo(self, kernel:DitherKernel, testname:str)->np.float64:
        """Tests a single kernel with a series of progressively more difficult
        images. Standard test for rectangular pixels."""
        idd = str(uuid.uuid4())
        os.mkdir(f"{PATH}{idd}")

        logging.info(f"BEGIN {testname} TEST AT {datetime.datetime.now()}")
        logging.info(f"UUID CODE is {idd}")
        logging.info(f"OUTPUT DIR is {idd}")

        score = []
        tempscore = self.test_ditheralgo(kernel, "David_tiny", idd, False, testname)
        score.append(tempscore)
        if (tempscore < 0.6): return score

        tempscore = self.test_ditheralgo(kernel, "astronaut", idd, False, testname)
        score.append(tempscore)
        if (tempscore < 0.6): return score

        tempscore = self.test_ditheralgo(kernel, "bricks", idd, False, testname)
        score.append(tempscore)
        if (tempscore < 0.5): return score

        tempscore = self.test_ditheralgo(kernel, "David", idd, False, testname)
        score.append(tempscore)
        if (tempscore < 0.5): return score

        logging.info(f"COMPLETE {testname} TEST AT {datetime.datetime.now()}")

        return score

    def getImage(self, path:str) -> np.ndarray:
        img = cv2.imread(path)

        if img is None:
            raise Exception("Could not find image with path " + path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img


def full_test_image(filename:str):
    """Tests a single image in a series of dithers and controls. This is used to determine
    whether it is a good image to use to test new dithers"""
    logging.info(f"BEGIN {filename} TEST AT {datetime.datetime.now()}")

    image = self.getImage(PATH + filename + ".png")
    d = Ditherer(image)
    imc = ImageCompare(d)

    imc.test_itself(image, filename)
    imc.test_random(image, filename)
    imc.test_cutoffs(image, filename)
    
    imc.run_test(imc.d.dither(DitherKernel("Floyd-Steinberg"), False), image, filename, "Floyd-Steinberg", False)
    imc.run_test(imc.d.dither(DitherKernel("Floyd-Steinberg"), True), image, filename, "Floyd-Steinberg", True)    
    imc.run_test(imc.d.dither(DitherKernel("Atkinson"), False), image, filename, "Atkinson", False)
    imc.run_test(imc.d.dither(DitherKernel("Atkinson"), True), image, filename, "Atkinson", True)

    logging.info(f"COMPLETE {filename} TEST AT {datetime.datetime.now()}")


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

    d = Ditherer(None)
    imc = ImageCompare(d)
    imc.greyscale = "data/07246.npy"
    imc.original_img = "out_img/07246.png"
#    score = imc.full_test_ditheralgo(DitherKernel("Temkin1"), "temkin1")
    score = imc.imgtest(DitherKernel("FloydSteinberg"), "test_tri", False, now)
    logging.info(f"score = {score}")
