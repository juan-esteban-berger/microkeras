from microkeras.operations.backward.calculate_dZ_wrapper import (
    calculate_dZ_wrapper
)
from microkeras.operations.backward.calculate_dW_wrapper import (
    calculate_dW_wrapper
)
from microkeras.operations.backward.calculate_db_wrapper import (
    calculate_db_wrapper
)

def backward_layer(model, layer_index, X, Y, A_prev, loss, m):
    """
    Perform backward propagation for a single layer in the model.

    Parameters:
    model (Sequential): The neural network model.
    layer_index (int): Index of the current layer.
    X (numpy.ndarray): Input data.
    Y (numpy.ndarray): True labels.
    A_prev (numpy.ndarray): Activation from the previous layer.
    loss (str): Loss function used.
    m (int): Number of training examples.

    Returns:
    tuple: (dZ, dW, db) gradients for the current layer.
    """
    layer = model.layers[layer_index]
    
    dZ = calculate_dZ_wrapper(model, layer_index, Y, loss)
    layer.dZ = dZ  # Set dZ for the current layer
    dW = calculate_dW_wrapper(model, layer_index, X if layer_index == 0 else A_prev, m)
    db = calculate_db_wrapper(model, layer_index, m)
    
    return dZ, dW, db
