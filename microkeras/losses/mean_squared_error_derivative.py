import numpy as np

def mean_squared_error_derivative(Y, Y_hat):
    """
    Calculate the derivative of the mean squared error loss.

    This function computes the gradient of the mean squared error
    with respect to the predicted values.

    Parameters:
    Y (numpy.ndarray): True values.
    Y_hat (numpy.ndarray): Predicted values.

    Returns:
    numpy.ndarray: The gradient of the mean squared error.

    Note:
    The result is divided by the number of samples (Y.shape[1])
    to get the average gradient across all samples.
    """
    return 2 * (Y_hat - Y) / Y.shape[1]
