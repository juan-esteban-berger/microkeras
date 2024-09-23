import numpy as np

def calculate_dZ_relu_mean_squared_error(A, Y, Z):
    m = A.shape[1]
    dA = 2 * (A - Y) / m
    dZ = dA * (Z > 0)
    return dZ
