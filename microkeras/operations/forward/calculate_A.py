from ...activations import sigmoid, softmax, relu, linear

def calculate_A(Z, activation):
    """
    Calculate the activation for a given input and activation function.

    Parameters:
    Z (numpy.ndarray): The input to the activation function.
    activation (str): The name of the activation function to use.
                      Options: 'sigmoid', 'softmax', 'relu', 'linear'

    Returns:
    numpy.ndarray: The output of the activation function.

    Raises:
    ValueError: If an unsupported activation function is specified.
    """
    if activation == 'sigmoid':
        return sigmoid(Z)
    elif activation == 'softmax':
        return softmax(Z)
    elif activation == 'relu':
        return relu(Z)
    elif activation == 'linear':
        return linear(Z)
