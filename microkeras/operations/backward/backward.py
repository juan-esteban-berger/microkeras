from .backward_layer import backward_layer

def backward(model, X, Y, loss):
    """
    Perform backward propagation through the entire model.

    Args:
        model (Sequential): The neural network model.
        X (numpy.ndarray): Input data.
        Y (numpy.ndarray): True labels.
        loss (str): Loss function used.

    Example:
        ```python
        backward(model, X_train, Y_train, 'categorical_crossentropy')
        ```

    Note:
        This function updates the gradients (dZ, dW, db) for all layers in the model.
    """
    m = X.shape[1]  # number of training examples
    n_layers = len(model.layers)
    
    for i in reversed(range(n_layers)):
        if i == n_layers - 1:
            A_prev = model.layers[i-1].A if i > 0 else X
        else:
            A_prev = model.layers[i-1].A
        
        dZ, dW, db = backward_layer(model, i, X, Y, A_prev, loss, m)
        
        model.layers[i].dZ = dZ
        model.layers[i].dW = dW
        model.layers[i].db = db
