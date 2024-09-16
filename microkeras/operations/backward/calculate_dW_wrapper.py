from microkeras.operations.backward.calculate_dW import calculate_dW

def calculate_dW_wrapper(model, i, X, m):
    current_layer = model.layers[i]
    
    if i != 0:
        previous_layer = model.layers[i-1]
        return calculate_dW(current_layer.dZ, previous_layer.A, m)
    else:
        return calculate_dW(current_layer.dZ, X, m)
