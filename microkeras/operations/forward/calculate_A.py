from ...activations import sigmoid, softmax

def calculate_A(Z, activation):
    if activation == 'sigmoid':
        return sigmoid(Z)
    elif activation == 'softmax':
        return softmax(Z)
