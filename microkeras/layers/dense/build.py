import numpy as np

def build(self, input_shape):
    """
    Build the dense layer by initializing weights and biases.

    This method initializes the weight matrix and bias vector for the dense layer
    based on the input shape and the number of units in the layer.

    ### Parameters
    - `input_shape` (int or tuple): The shape of the input to this layer. If tuple, the first element is used.

    ### Notes
    - **Weights** are initialized randomly in the range [-0.5, 0.5].
    - **Biases** are initialized randomly in the range [-0.5, 0.5].

    ### Example
    ```python
    layer = Dense(64, activation='relu')
    layer.build(input_shape=(128,))
    print(layer.W.shape)  # Output: (64, 128)
    print(layer.b.shape)  # Output: (64, 1)
    ```

    ### Attributes Set
    - `self.W`: Weight matrix of shape `(self.units, self.input_shape)`
    - `self.b`: Bias vector of shape `(self.units, 1)`
    """
    if isinstance(input_shape, tuple):
        self.input_shape = input_shape[0]
    else:
        self.input_shape = input_shape
    self.W = np.random.rand(self.units, self.input_shape) - 0.5
    self.b = np.random.rand(self.units, 1) - 0.5
