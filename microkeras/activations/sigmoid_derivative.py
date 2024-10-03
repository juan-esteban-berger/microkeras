import numpy as np
from .sigmoid import sigmoid

def sigmoid_derivative(Z):
    """
    Compute the derivative of the sigmoid activation function.

    Args:
        Z (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: An array with the same shape as Z, containing the sigmoid derivative values.

    Example:
        ```python
        Z = np.array([[-1, 0], [1, 2]])
        dZ = sigmoid_derivative(Z)
        print(dZ)
        ```
    """
    activation = sigmoid(Z)
    return activation * (1 - activation)
