import numpy as np
from .sigmoid import sigmoid

def sigmoid_derivative(Z):
    """
    Compute the derivative of the sigmoid activation function.

    The derivative of the sigmoid function is f'(x) = f(x) * (1 - f(x)),
    where f(x) is the sigmoid function.

    Parameters:
    Z (numpy.ndarray): The input array.

    Returns:
    numpy.ndarray: An array with the same shape as Z, containing the sigmoid derivative values.
    """
    activation = sigmoid(Z)
    return activation * (1 - activation)
