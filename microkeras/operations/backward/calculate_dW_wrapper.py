from microkeras.operations.backward.calculate_dW import calculate_dW

def calculate_dW_wrapper(model, i, X, m):
    """
    Wrapper function to calculate the gradient of the weights for a specific layer.

    Args:
        model (Sequential): The neural network model.
        i (int): Index of the current layer.
        X (numpy.ndarray): Input data (used for the first layer).
        m (int): Number of training examples.

    Returns:
        numpy.ndarray: Gradient of the weights for the specified layer.

    Example:
        ```python
        dW = calculate_dW_wrapper(model, 1, X, 32)
        print(dW.shape)
        ```
    """
    current_layer = model.layers[i]
    
    if i != 0:
        previous_layer = model.layers[i-1]
        return calculate_dW(current_layer.dZ, previous_layer.A, m)
    else:
        return calculate_dW(current_layer.dZ, X, m)
