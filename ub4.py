from PIL import Image, ImageDraw
from random import randrange

OUT_FILE = "out_img/ub4_2.png"

class Pattern:

    def __init__(self):
        self.scale = 100
        self.repeat_square = 3.95
        self.size = 1500
        self.colors = [(255,255,0,0), (0,0,255,0), (0,255,0,0), (255,0,0,0),(255,0,255,0),(0,255,255,0)]

        self.__curr_color = -1 # should be reset at beginning of draw()

    def rand_color(self):
        return self.colors[randrange(len(self.colors))]

    def cycle_color(self):
        self.__curr_color += 1
        if (self.__curr_color >= len(self.colors)):
            self.__curr_color = 0
        return self.colors[self.__curr_color]

    def get_color(self):
        return self.cycle_color()
        # return self.rand_color()

    def draw_polygon(self, polygon, drawer, color, x, y):
        polygon = list(map(lambda c: (
            (c[0] + x * self.repeat_square) * self.scale, 
            (c[1] + y * self.repeat_square) * self.scale), polygon))
        drawer.polygon(polygon, fill=color)

    def draw(self):
        self.curr_color = -1
        im = Image.new('CMYK', (self.size, self.size),(255,255,255,255))
        drawer = ImageDraw.Draw(im)

        for y in range(-1, 50):
            for x in range(-1, 50):
                topsquare = [(0.848, 0),(1.378, 0.848),(0.53, 1.378),(0, 0.53)]
                self.draw_polygon(topsquare, drawer, self.get_color(), x, y)

                first_heptagon = [(0.53, 1.378),(1.378, 0.848),(2.34, 1.05),(2.5,2),(2,2.8),(1,3),(0.34,2.33)]
                self.draw_polygon(first_heptagon, drawer, self.get_color(), x, y)

                second_heptagon = [(2.5,2),(2.34, 1.05),(3,0.35),(3.95,0.53),(4.48,1.378),(4.28,2.34),(3.33,2.5)]
                self.draw_polygon(second_heptagon, drawer, self.get_color(), x, y)

                middlesquare = [(2,2.8),(2.5,2),(3.34,2.5),(2.8,3.33)]
                self.draw_polygon(middlesquare, drawer, self.get_color(), x, y)

                third_heptagon = [(2.8,3.33),(3.34,2.5),(4.28,2.34),(4.97,3),(4.76,3.97),(3.94,4.5),(2.98,4.3)]
                self.draw_polygon(third_heptagon, drawer, self.get_color(), x, y)

                fourth_heptagon = [(2.8,3.35),(3,4.3),(2.32,5.02),(1.35,4.8),(0.83,3.96),(1,3),(2,2.8)]
                self.draw_polygon(fourth_heptagon, drawer, self.get_color(), x, y)

        if im.mode == 'CMYK':
            im = im.convert('RGB')

        im.save(OUT_FILE)

x = Pattern()
#x.colors = [(0,0,0,0),(0,0,0,255)] # black and white
x.size = 15000
x.draw()
