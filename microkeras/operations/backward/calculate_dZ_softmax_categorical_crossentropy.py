import numpy as np

def calculate_dZ_softmax_categorical_crossentropy(A, Y):
    """
    Calculate dZ for softmax activation with categorical cross-entropy loss.

    Args:
        A (numpy.ndarray): Activations of the current layer (softmax output).
        Y (numpy.ndarray): True labels (one-hot encoded).

    Returns:
        numpy.ndarray: Gradient of Z.

    Example:
        ```python
        dZ = calculate_dZ_softmax_categorical_crossentropy(A, Y)
        print(dZ.shape)
        ```
    """
    return A - Y
