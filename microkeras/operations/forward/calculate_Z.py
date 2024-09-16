import numpy as np

def calculate_Z(W, A_prev, b):
    return np.dot(W, A_prev) + b
