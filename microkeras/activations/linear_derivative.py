import numpy as np

def linear_derivative(Z):
    """
    Compute the derivative of the linear activation function.

    The derivative of the linear function is always 1, so this function
    returns an array of ones with the same shape as the input.

    Parameters:
    Z (numpy.ndarray): The input array.

    Returns:
    numpy.ndarray: An array of ones with the same shape as Z.
    """
    return np.ones_like(Z)
