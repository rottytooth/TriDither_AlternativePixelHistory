## Tri-Dither 

A collection of dithering programs

# Ditherer: takes an image or numpy array of grey values [each 0 to 255] and dithers to black and white
- takes argument for percentage dithering
- argument for serpentine dither
- returns new set of values (each 0 or 255), does not write out image
- constructor takes the image or array we're working on

# Shapizer: pixelates an image according to the given shape. Optionally shapes the entire image to that shape as well (for some shapes, that always happens)
- saves a numpy array of the pixelated colors, so the image doesn't need to be reread each time we are to dither. The 
Shapizers:
    - Triangularizer
    - Righttriangularizer

# Triangularize_test: A script to dither a triangular shape that has already been "shapized". Can be set for triangle or right-triangle

Essentially, it's like running the ditherer except it writes out the image

