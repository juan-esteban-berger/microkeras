import numpy as np
from ...activations import sigmoid_derivative, relu_derivative, linear_derivative

def calculate_dZ(W_next, dZ_next, Z, activation):
    """
    Calculate dZ for hidden layers.

    Parameters:
    W_next (numpy.ndarray): Weights of the next layer.
    dZ_next (numpy.ndarray): dZ of the next layer.
    Z (numpy.ndarray): Linear output of the current layer.
    activation (str): Activation function of the current layer.

    Returns:
    numpy.ndarray: Gradient of Z.
    """
    activation_derivatives = {
        'sigmoid': sigmoid_derivative,
        'relu': relu_derivative,
        'linear': linear_derivative
    }
    activation_derivative = activation_derivatives[activation]
    return np.dot(W_next.T, dZ_next) * activation_derivative(Z)
