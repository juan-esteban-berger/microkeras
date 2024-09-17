import copy as cp

def copy(self):
    new_layer = self.__class__(self.units,
                               activation=self.activation,
                               input_shape=self.input_shape)
    if self.W is not None:
        new_layer.W = cp.deepcopy(self.W)
    if self.b is not None:
        new_layer.b = cp.deepcopy(self.b)
    return new_layer
