import numpy as np

class Dense:
    def __init__(self, units, activation=None, input_shape=None):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.batch_size = None
        self.W = np.random.rand(units, input_shape) - 0.5
        self.b = np.random.rand(units, 1) - 0.5
        self.Z = None
        self.A = None
        self.dZ = None
        self.dW = None
        self.db = None
