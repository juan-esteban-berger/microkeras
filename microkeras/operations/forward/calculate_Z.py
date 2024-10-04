import numpy as np

def calculate_Z(W, A_prev, b):
    """
    Calculate the linear combination Z = W * A_prev + b.

    Args:
        W (numpy.ndarray): Weight matrix of the current layer.
        A_prev (numpy.ndarray): Activation output from the previous layer.
        b (numpy.ndarray): Bias vector of the current layer.

    Returns:
        numpy.ndarray: The linear combination Z.

    Example:
        ```python
        W = np.random.randn(3, 4)
        A_prev = np.random.randn(4, 5)
        b = np.random.randn(3, 1)
        Z = calculate_Z(W, A_prev, b)
        print(Z.shape)
        ```
    """
    return np.dot(W, A_prev) + b
