import numpy as np

def calculate_dW(dZ, A_prev, m):
    """
    Calculate the gradient of the weights.

    Parameters:
    dZ (numpy.ndarray): Gradient of the cost with respect to the linear output.
    A_prev (numpy.ndarray): Activations from the previous layer.
    m (int): Number of training examples.

    Returns:
    numpy.ndarray: Gradient of the weights.
    """
    result = (1 / m) * np.dot(dZ, A_prev.T) 
    return np.array(result).astype(np.float64)
