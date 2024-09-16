from .backward_layer import backward_layer

def backward(model, X, Y, loss):
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
