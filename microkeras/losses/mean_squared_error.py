import numpy as np

def mean_squared_error(Y, Y_hat):
    return np.mean((Y - Y_hat) ** 2)
