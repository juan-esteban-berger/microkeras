from .initialize import initialize
from .build import build
from .copy import copy

class Dense:
    """
    Represents a dense (fully connected) layer in a neural network.

    This class implements a dense layer with configurable number of units and activation function.

    Attributes:
    units (int): The number of neurons in the layer.
    activation (str or None): The activation function to use.
    input_shape (int or tuple): The shape of the input to this layer.

    Methods:
    build(input_shape): Initialize the layer's weights and biases.
    copy(): Create a deep copy of the layer.
    """
    def __init__(self, units, activation=None, input_shape=None):
        """
        Initialize the dense layer.

        Parameters:
        units (int): The number of neurons in the layer.
        activation (str or None): The activation function to use. Defaults to None.
        input_shape (int or tuple): The shape of the input to this layer. Defaults to None.
        """
        initialize(self, units, activation, input_shape)

    build = build
    copy = copy
