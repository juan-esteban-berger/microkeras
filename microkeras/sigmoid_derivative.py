import numpy as np
from .sigmoid import sigmoid

def sigmoid_derivative(Z):
    activation = sigmoid(Z)
    return activation * (1 - activation)
