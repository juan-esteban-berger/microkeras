import numpy as np

def mean_squared_error(Y, Y_hat):
    """
    Calculate the mean squared error loss.

    Args:
        Y (numpy.ndarray): True values.
        Y_hat (numpy.ndarray): Predicted values.

    Returns:
        float: The mean squared error between Y and Y_hat.

    Example:
        ```python
        Y = np.array([[1, 2], [3, 4]])
        Y_hat = np.array([[1.1, 2.1], [2.9, 4.1]])
        mse = mean_squared_error(Y, Y_hat)
        print(mse)
        ```

    Note:
        This function computes the average squared difference
        between the true values and the predicted values.
    """
    return np.mean((Y - Y_hat) ** 2)
