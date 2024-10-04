def update_params(model, learning_rate):
    """
    Update the parameters (weights and biases) of all layers in the model.

    Args:
        model (Sequential): The neural network model.
        learning_rate (float): The learning rate for gradient descent.

    Example:
        ```python
        update_params(model, 0.01)
        ```

    Note:
        This function applies the computed gradients to update the model's parameters.
    """
    for layer in model.layers:
        layer.W -= learning_rate * layer.dW
        layer.b -= learning_rate * layer.db
