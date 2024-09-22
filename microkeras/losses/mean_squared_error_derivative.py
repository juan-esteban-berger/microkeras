import numpy as np

def mean_squared_error_derivative(Y, Y_hat):
    return 2 * (Y_hat - Y) / Y.shape[1]
