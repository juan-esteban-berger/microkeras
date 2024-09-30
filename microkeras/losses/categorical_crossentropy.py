import numpy as np

def categorical_crossentropy(Y, Y_hat):
    """
    Calculate the categorical cross-entropy loss.

    Categorical cross-entropy is commonly used in multi-class classification problems.

    Parameters:
    Y (numpy.ndarray): True labels in one-hot encoded format.
    Y_hat (numpy.ndarray): Predicted probabilities for each class.

    Returns:
    float: The categorical cross-entropy loss.

    Note:
    A small epsilon (1e-8) is added to the log to avoid numerical instability
    when Y_hat contains zero probabilities.
    """
    return -np.sum(Y * np.log(Y_hat + 1e-8))
