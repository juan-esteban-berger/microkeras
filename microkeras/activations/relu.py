import numpy as np

def relu(Z):
    """
    Compute the ReLU (Rectified Linear Unit) activation function.

    ReLU returns the input for all positive values, and 0 for all non-positive values.

    Parameters:
    Z (numpy.ndarray): The input array.

    Returns:
    numpy.ndarray: An array with the same shape as Z, containing the ReLU activation values.
    """
    return np.maximum(0, Z)
