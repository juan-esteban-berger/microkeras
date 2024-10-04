import numpy as np

def mean_squared_error_derivative(Y, Y_hat):
    """
    Calculate the derivative of the mean squared error loss.

    Args:
        Y (numpy.ndarray): True values.
        Y_hat (numpy.ndarray): Predicted values.

    Returns:
        numpy.ndarray: The gradient of the mean squared error.

    Example:
        ```python
        Y = np.array([[1, 2], [3, 4]])
        Y_hat = np.array([[1.1, 2.1], [2.9, 4.1]])
        gradient = mean_squared_error_derivative(Y, Y_hat)
        print(gradient)
        ```

    Note:
        The result is divided by the number of samples (Y.shape[1])
        to get the average gradient across all samples.
    """
    return 2 * (Y_hat - Y) / Y.shape[1]
