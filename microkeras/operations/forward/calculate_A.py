from ...activations import sigmoid, softmax, relu

def calculate_A(Z, activation):
    if activation == 'sigmoid':
        return sigmoid(Z)
    elif activation == 'softmax':
        return softmax(Z)
    elif activation == 'relu':
        return relu(Z)
