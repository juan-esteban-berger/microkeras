import numpy as np
from microkeras.operations.forward import forward

def get_accuracy(model, X, Y):
    """
    Calculate the accuracy of the model's predictions.

    This function performs a forward pass on the input data and compares
    the model's predictions to the true labels to compute the accuracy.

    Parameters:
    model (Sequential): The neural network model.
    X (numpy.ndarray): Input data, shape (n_features, n_samples).
    Y (numpy.ndarray): True labels in one-hot encoded format, shape (n_classes, n_samples).

    Returns:
    float: The accuracy of the model's predictions, as a value between 0 and 1.

    Note:
    - This function assumes that the model's output and true labels are in one-hot encoded format.
    - The accuracy is calculated as the ratio of correct predictions to total predictions.
    """
    forward(model, X)
    predictions = np.argmax(model.layers[-1].A, axis=0)
    Y_decoded = np.argmax(Y, axis=0)
    return np.sum(predictions == Y_decoded) / Y_decoded.size
