from .initialize import initialize
from .build import build
from .copy import copy

class Dense:
    """
    Represents a dense (fully connected) layer in a neural network.

    Attributes:
        units (int): The number of neurons in the layer.
        activation (str or None): The activation function to use.
        input_shape (int or tuple): The shape of the input to this layer.

    Example:
        ```python
        # Create a dense layer with 64 units and ReLU activation
        layer = Dense(64, activation='relu', input_shape=(128,))
        
        # Build the layer
        layer.build(input_shape=(128,))
        
        # Use the layer in a model
        x = np.random.randn(32, 128)  # Batch of 32 samples
        output = layer.forward(x)
        print(output.shape)  # Output: (32, 64)
        ```
    """
    def __init__(self, units, activation=None, input_shape=None):
        """
        Initialize the dense layer.

        Args:
            units (int): The number of neurons in the layer.
            activation (str or None): The activation function to use. Default is None.
            input_shape (int or tuple): The shape of the input to this layer. Default is None.
        """
        initialize(self, units, activation, input_shape)

    build = build
    copy = copy
