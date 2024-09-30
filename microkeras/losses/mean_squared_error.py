import numpy as np

def mean_squared_error(Y, Y_hat):
    """
    Calculate the mean squared error loss.

    Mean Squared Error (MSE) is commonly used in regression problems.

    Parameters:
    Y (numpy.ndarray): True values.
    Y_hat (numpy.ndarray): Predicted values.

    Returns:
    float: The mean squared error between Y and Y_hat.

    Note:
    This function computes the average squared difference
    between the true values and the predicted values.
    """
    return np.mean((Y - Y_hat) ** 2)
