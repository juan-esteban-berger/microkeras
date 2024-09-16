import numpy as np

def calculate_dW(dZ, A_prev, m):
    result = (1 / m) * np.dot(dZ, A_prev.T) 
    return np.array(result).astype(np.float64)
