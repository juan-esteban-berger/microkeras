def update_params(model, learning_rate):
    for layer in model.layers:
        layer.W -= learning_rate * layer.dW
        layer.b -= learning_rate * layer.db
