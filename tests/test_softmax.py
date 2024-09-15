import numpy as np

def softmax(Z):
    exp_shifted = np.exp(Z - np.max(Z))
    return exp_shifted / (exp_shifted.sum(axis=0) + 1e-8)
