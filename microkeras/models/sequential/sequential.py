from .initialize import initialize
from .add import add
from .build import build

class Sequential:
    def __init__(self, layers):
        initialize(self, layers)

    add = add
    build = build
