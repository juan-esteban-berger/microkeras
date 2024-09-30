import numpy as np

def calculate_Z(W, A_prev, b):
    """
    Calculate the linear combination Z = W * A_prev + b.

    Parameters:
    W (numpy.ndarray): Weight matrix of the current layer.
    A_prev (numpy.ndarray): Activation output from the previous layer.
    b (numpy.ndarray): Bias vector of the current layer.

    Returns:
    numpy.ndarray: The linear combination Z.
    """
    return np.dot(W, A_prev) + b
