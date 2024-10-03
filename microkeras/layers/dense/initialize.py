def initialize(self, units, activation, input_shape):
    """
    Initialize the attributes of the dense layer.

    Args:
        units (int): The number of neurons in the layer.
        activation (str or None): The activation function to use.
        input_shape (int or tuple): The shape of the input to this layer.

    Example:
        ```python
        layer = Dense(64)
        print(layer.units)  # Output: 64
        print(layer.activation)  # Output: None
        print(layer.W)  # Output: None (not yet built)
        
        layer.build(input_shape=(128,))
        print(layer.W.shape)  # Output: (64, 128)
        print(layer.b.shape)  # Output: (64, 1)
        ```
    """
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
