import numpy as np

def categorical_crossentropy(Y, Y_hat):
    """
    Calculate the categorical cross-entropy loss.

    Args:
        Y (numpy.ndarray): True labels in one-hot encoded format.
        Y_hat (numpy.ndarray): Predicted probabilities for each class.

    Returns:
        float: The categorical cross-entropy loss.

    Example:
        ```python
        Y = np.array([[1, 0], [0, 1]])
        Y_hat = np.array([[0.9, 0.1], [0.2, 0.8]])
        loss = categorical_crossentropy(Y, Y_hat)
        print(loss)
        ```

    Note:
        A small epsilon (1e-8) is added to the log to avoid numerical instability
        when Y_hat contains zero probabilities.
    """
    return -np.sum(Y * np.log(Y_hat + 1e-8))
