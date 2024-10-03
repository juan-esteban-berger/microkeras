import numpy as np

def relu_derivative(Z):
    """
    Compute the derivative of the ReLU (Rectified Linear Unit) activation function.

    Args:
        Z (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: An array with the same shape as Z, containing 1.0 where Z > 0 and 0.0 elsewhere.

    Example:
        ```python
        Z = np.array([[-1, 0], [1, 2]])
        dZ = relu_derivative(Z)
        print(dZ)
        ```
    """
    return np.where(Z > 0, 1.0, 0.0)
