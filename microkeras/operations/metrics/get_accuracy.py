import numpy as np

def get_accuracy(model, X, Y):
    predictions = np.argmax(model.layers[-1].A, axis=0)
    Y_decoded = np.argmax(Y, axis=0)
    return np.sum(predictions == Y_decoded) / Y_decoded.size
