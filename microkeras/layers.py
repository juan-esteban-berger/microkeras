import numpy as np

class Dense:
    def __init__(self, units, activation=None, input_shape=None):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.batch_size = None
        self.W = None
        self.b = None
        self.Z = None
        self.A = None
        self.dZ = None
        self.dW = None
        self.db = None

    def build(self, input_shape):
        if isinstance(input_shape, tuple):
            self.input_shape = input_shape[0]
        else:
            self.input_shape = input_shape
        self.W = np.random.rand(self.units, self.input_shape) - 0.5
        self.b = np.random.rand(self.units, 1) - 0.5
