import numpy as np

def relu(Z):
    """
    Compute the ReLU (Rectified Linear Unit) activation function.

    Args:
        Z (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: An array with the same shape as Z, containing the ReLU activation values.

    Example:
        ```python
        Z = np.array([[-1, 0], [1, 2]])
        A = relu(Z)
        print(A)
        ```
    """
    return np.maximum(0, Z)
