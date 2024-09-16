import numpy as np

def categorical_crossentropy(Y, Y_hat):
    return -np.sum(Y * np.log(Y_hat + 1e-8))
