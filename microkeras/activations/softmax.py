import numpy as np

def softmax(Z):
    """
    Compute the softmax activation function.

    The softmax function is defined as exp(x_i) / sum(exp(x_j)) for all j.
    This implementation uses the shifted softmax for numerical stability.

    Parameters:
    Z (numpy.ndarray): The input array.

    Returns:
    numpy.ndarray: An array with the same shape as Z, containing the softmax activation values.
    """
    exp_shifted = np.exp(Z - np.max(Z))
    return exp_shifted / (exp_shifted.sum(axis=0) + 1e-8)
