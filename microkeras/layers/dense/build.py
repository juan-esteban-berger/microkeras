import numpy as np

def build(self, input_shape):
    """
    Build the dense layer by initializing weights and biases.

    Args:
        input_shape (int or tuple): The shape of the input to this layer. If tuple, the first element is used.

    Example:
        ```python
        layer = Dense(64, activation='relu')
        layer.build(input_shape=(128,))
        print(layer.W.shape)  # Output: (64, 128)
        print(layer.b.shape)  # Output: (64, 1)
        ```
    """
    if isinstance(input_shape, tuple):
        self.input_shape = input_shape[0]
    else:
        self.input_shape = input_shape
    self.W = np.random.rand(self.units, self.input_shape) - 0.5
    self.b = np.random.rand(self.units, 1) - 0.5
