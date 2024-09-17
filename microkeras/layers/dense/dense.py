from .initialize import initialize
from .build import build
from .copy import copy

class Dense:
    def __init__(self, units, activation=None, input_shape=None):
        initialize(self, units, activation, input_shape)

    build = build
    copy = copy
