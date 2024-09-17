from .initialize import initialize
from .minimize import minimize

class SGD:
    def __init__(self, learning_rate=0.01):
        initialize(self, learning_rate)
    minimize = minimize
