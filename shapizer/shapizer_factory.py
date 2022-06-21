import logging
from shapizer import shapizer
from .hexagonizer import Hexagonizer
from .triangularizer import Triangularizer
from .righttriangularizer import Righttriangularizer

class ShapizerFactory():
    @staticmethod
    def GetShapizer(shape:str, args=None) -> shapizer.Shapizer:
        if shape == "triangle":
            shapizer = Triangularizer(args)
        if shape == "righttriangle":
            shapizer = Righttriangularizer(args)
        if shape == "hexagon":
            shapizer = Hexagonizer(args)

        return shapizer