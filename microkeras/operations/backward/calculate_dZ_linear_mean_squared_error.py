import numpy as np

def calculate_dZ_linear_mean_squared_error(A, Y, Z):
    m = A.shape[1]
    dZ = 2 * (A - Y) / m
    return dZ
