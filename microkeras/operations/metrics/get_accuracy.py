import numpy as np
from microkeras.operations.forward import forward

def get_accuracy(model, X, Y):
    forward(model, X)
    predictions = np.argmax(model.layers[-1].A, axis=0)
    Y_decoded = np.argmax(Y, axis=0)
    return np.sum(predictions == Y_decoded) / Y_decoded.size
