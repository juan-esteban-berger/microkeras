import numpy as np
from ...activations import sigmoid_derivative, relu_derivative

def calculate_dZ(W_next, dZ_next, Z, activation):
    activation_derivatives = {
        'sigmoid': sigmoid_derivative,
        'relu': relu_derivative
    }
    activation_derivative = activation_derivatives[activation]
    return np.dot(W_next.T, dZ_next) * activation_derivative(Z)
