from ...activations import sigmoid, softmax, relu, linear

def calculate_A(Z, activation):
    if activation == 'sigmoid':
        return sigmoid(Z)
    elif activation == 'softmax':
        return softmax(Z)
    elif activation == 'relu':
        return relu(Z)
    elif activation == 'linear':
        return linear(Z)
