def update_params(model, learning_rate):
    """
    Update the parameters (weights and biases) of all layers in the model.

    Parameters:
    model (Sequential): The neural network model.
    learning_rate (float): The learning rate for gradient descent.

    This function applies the computed gradients to update the model's parameters.
    """
    for layer in model.layers:
        layer.W -= learning_rate * layer.dW
        layer.b -= learning_rate * layer.db
