import numpy as np

def softmax(Z):
    """
    Compute the softmax activation function.

    Args:
        Z (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: An array with the same shape as Z, containing the softmax activation values.

    Example:
        ```python
        Z = np.array([[1, 2], [3, 4]])
        A = softmax(Z)
        print(A)  # Output: [[0.119203 0.119203] [0.880797 0.880797]]
        ```
    """
    exp_shifted = np.exp(Z - np.max(Z))
    return exp_shifted / (exp_shifted.sum(axis=0) + 1e-8)
