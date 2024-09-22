import numpy as np

def relu_derivative(Z):
    return np.where(Z > 0, 1.0, 0.0)
