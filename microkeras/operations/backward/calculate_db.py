import numpy as np

def calculate_db(dZ, m):
    """
    Calculate the gradient of the bias.

    Args:
        dZ (numpy.ndarray): Gradient of the cost with respect to the linear output.
        m (int): Number of training examples.

    Returns:
        numpy.ndarray: Gradient of the bias.

    Example:
        ```python
        db = calculate_db(dZ, 32)
        print(db.shape)
        ```
    """
    return (1 / m) * np.sum(dZ, axis=1, keepdims=True)
