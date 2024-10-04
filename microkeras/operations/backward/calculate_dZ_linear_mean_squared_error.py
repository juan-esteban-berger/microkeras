import numpy as np

def calculate_dZ_linear_mean_squared_error(A, Y, Z):
    """
    Calculate dZ for linear activation with mean squared error loss.

    Args:
        A (numpy.ndarray): Activations of the current layer.
        Y (numpy.ndarray): True labels.
        Z (numpy.ndarray): Linear output of the current layer.

    Returns:
        numpy.ndarray: Gradient of Z.

    Example:
        ```python
        dZ = calculate_dZ_linear_mean_squared_error(A, Y, Z)
        print(dZ.shape)
        ```
    """
    m = A.shape[1]
    dZ = 2 * (A - Y) / m
    return dZ
