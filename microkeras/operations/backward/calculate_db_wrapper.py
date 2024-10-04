from microkeras.operations.backward.calculate_db import calculate_db

def calculate_db_wrapper(model, i, m):
    """
    Wrapper function to calculate the gradient of the bias for a specific layer.

    Args:
        model (Sequential): The neural network model.
        i (int): Index of the current layer.
        m (int): Number of training examples.

    Returns:
        numpy.ndarray: Gradient of the bias for the specified layer.

    Example:
        ```python
        db = calculate_db_wrapper(model, 1, 32)
        print(db.shape)
        ```
    """
    return calculate_db(model.layers[i].dZ, m)
