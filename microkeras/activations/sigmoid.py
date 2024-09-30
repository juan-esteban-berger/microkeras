import numpy as np

def sigmoid(Z):
    """
    Compute the sigmoid activation function.

    The sigmoid function is defined as f(x) = 1 / (1 + e^(-x)).

    Parameters:
    Z (numpy.ndarray): The input array.

    Returns:
    numpy.ndarray: An array with the same shape as Z, containing the sigmoid activation values.
    """
    return 1 / (1 + np.exp(-Z))
