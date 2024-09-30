import numpy as np

def relu_derivative(Z):
    """
    Compute the derivative of the ReLU (Rectified Linear Unit) activation function.

    The derivative of ReLU is 1 for all positive values, and 0 for all non-positive values.

    Parameters:
    Z (numpy.ndarray): The input array.

    Returns:
    numpy.ndarray: An array with the same shape as Z, containing 1.0 where Z > 0 and 0.0 elsewhere.
    """
    return np.where(Z > 0, 1.0, 0.0)
