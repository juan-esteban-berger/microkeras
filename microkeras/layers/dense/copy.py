import copy as cp

def copy(self):
    """
    Create a deep copy of the dense layer.

    This method creates a new instance of the dense layer with the same configuration
    and performs a deep copy of the weights and biases if they have been initialized.

    Returns:
    Dense: A new Dense layer instance with copied attributes and parameters.
    """
    new_layer = self.__class__(self.units,
                               activation=self.activation,
                               input_shape=self.input_shape)
    if self.W is not None:
        new_layer.W = cp.deepcopy(self.W)
    if self.b is not None:
        new_layer.b = cp.deepcopy(self.b)
    return new_layer
