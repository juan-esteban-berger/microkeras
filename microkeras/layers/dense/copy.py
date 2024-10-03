import copy as cp

def copy(self):
    """
    Create a deep copy of the dense layer.

    This method creates a new instance of the dense layer with the same configuration
    and performs a deep copy of the weights and biases if they have been initialized.

    ### Returns
    `Dense`: A new `Dense` layer instance with copied attributes and parameters.

    ### Example
    ```python
    original_layer = Dense(32, activation='sigmoid', input_shape=(64,))
    original_layer.build(input_shape=(64,))
    
    copied_layer = original_layer.copy()
    print(np.array_equal(original_layer.W, copied_layer.W))  # Output: True
    print(id(original_layer.W) != id(copied_layer.W))  # Output: True
    ```

    ### Implementation Details
    - Uses `copy.deepcopy()` for `self.W` and `self.b`
    - Creates a new instance of the same class (`self.__class__`)
    - Copies all initialized attributes
    """
    new_layer = self.__class__(self.units,
                               activation=self.activation,
                               input_shape=self.input_shape)
    if self.W is not None:
        new_layer.W = cp.deepcopy(self.W)
    if self.b is not None:
        new_layer.b = cp.deepcopy(self.b)
    return new_layer
