import numpy as np

def calculate_dZ_relu_mean_squared_error(A, Y, Z):
    """
    Calculate dZ for ReLU activation with mean squared error loss.

    Parameters:
    A (numpy.ndarray): Activations of the current layer.
    Y (numpy.ndarray): True labels.
    Z (numpy.ndarray): Linear output of the current layer.

    Returns:
    numpy.ndarray: Gradient of Z.
    """
    m = A.shape[1]
    dA = 2 * (A - Y) / m
    dZ = dA * (Z > 0)
    return dZ
