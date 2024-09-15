import numpy as np

def build(self, input_shape):
    if isinstance(input_shape, tuple):
        self.input_shape = input_shape[0]
    else:
        self.input_shape = input_shape
    self.W = np.random.rand(self.units, self.input_shape) - 0.5
    self.b = np.random.rand(self.units, 1) - 0.5
