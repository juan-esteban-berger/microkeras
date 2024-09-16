import numpy as np

def calculate_db(dZ, m):
    return (1 / m) * np.sum(dZ, axis=1, keepdims=True)
