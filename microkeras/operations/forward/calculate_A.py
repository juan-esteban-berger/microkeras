from ...activations import sigmoid, softmax, relu, linear

def calculate_A(Z, activation):
    """
    Calculate the activation output for a given input and activation function.

    Args:
        Z (numpy.ndarray): The input to the activation function.
        activation (str): The name of the activation function to use.

    Returns:
        numpy.ndarray: The output after applying the activation function.

    Example:
        ```python
        Z = np.array([[1, 2], [3, 4]])
        A = calculate_A(Z, 'sigmoid')
        print(A.shape)
        ```
    """
    if activation == 'sigmoid':
        return sigmoid(Z)
    elif activation == 'softmax':
        return softmax(Z)
    elif activation == 'relu':
        return relu(Z)
    elif activation == 'linear':
        return linear(Z)
