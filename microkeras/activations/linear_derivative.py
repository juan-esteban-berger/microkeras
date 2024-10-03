import numpy as np

def linear_derivative(Z):
    """
    Compute the derivative of the linear activation function.

    Args:
        Z (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: An array of ones with the same shape as Z.

    Example:
        ```python
        Z = np.array([[1, 2], [3, 4]])
        dZ = linear_derivative(Z)
        print(dZ)
        ```
    """
    return np.ones_like(Z)
