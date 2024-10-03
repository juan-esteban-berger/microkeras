import copy as cp

def copy(self):
    """
    Create a deep copy of the dense layer.

    Returns:
        Dense: A new Dense layer instance with copied attributes and parameters.

    Example:
        ```python
        original_layer = Dense(32, activation='sigmoid', input_shape=(64,))
        original_layer.build(input_shape=(64,))
        
        copied_layer = original_layer.copy()
        print(np.array_equal(original_layer.W, copied_layer.W))  # Output: True
        print(id(original_layer.W) != id(copied_layer.W))  # Output: True
        ```
    """
    new_layer = self.__class__(self.units,
                               activation=self.activation,
                               input_shape=self.input_shape)
    if self.W is not None:
        new_layer.W = cp.deepcopy(self.W)
    if self.b is not None:
        new_layer.b = cp.deepcopy(self.b)
    return new_layer
