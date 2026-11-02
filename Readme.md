## Tri-Dither 

A collection of dithering programs

Created by [Daniel Temkin](https://danieltemkin.com)

# Ditherer: takes an image or numpy array of greyscale values [each an np.uint8 from 0 to 255] and dithers to black and white
- takes argument for percentage dithering
- argument for serpentine dither
- returns new set of values (each black or white, 0 or 255), does not write out image
- constructor takes the image or array we're working on
- has no concept of pixel shape

# Shapizer: pixelates an image according to the given shape. Optionally shapes the entire image to that shape as well (for some shapes, that always happens)
- saves a numpy array of the pixelated colors, so the image doesn't need to be reread each time we are to dither
Shapizers:
    - Triangularizer
    - Righttriangularizer
    - Hexagonizer

# Triangularize_test: A script to dither a triangular shape that has already been "shapized". Can be set for triangle or right-triangle

Essentially, it's like running the ditherer except it writes out the image

# ml_model / DitherTester and DitherTesterTriangle

- runs simulated annealing to generate dithers and then calculates ImageScore for an image

# image_compare / ImageCompare

Called directly, scores an image using Structural Similarity Index

- to do: add contrast controls so Atkinson dither, and others that are good for middle range, will test better


Requires Python 3.13

Has actual tests now. Run from this folder with `python -m unittest discover -s tests -t .`
Test files are written to `../out_test` (one folder per test), which is cleared at the start of each run. Set `TRIDITHER_TEST_OUT` to write them elsewhere.