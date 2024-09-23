from microkeras.operations.backward.calculate_dZ import calculate_dZ
from microkeras.operations.backward.calculate_dZ_softmax_categorical_crossentropy import (
    calculate_dZ_softmax_categorical_crossentropy
)
from microkeras.operations.backward.calculate_dZ_relu_mean_squared_error import (
    calculate_dZ_relu_mean_squared_error
)

def calculate_dZ_wrapper(model, i, Y, loss):
    current_layer = model.layers[i]
    
    if (i == len(model.layers) - 1):
        if (current_layer.activation == 'softmax' and
            loss == 'categorical_crossentropy'):
            return calculate_dZ_softmax_categorical_crossentropy(current_layer.A, Y)
        elif (current_layer.activation == 'relu' and
              loss == 'mean_squared_error'):
            return calculate_dZ_relu_mean_squared_error(current_layer.A, Y, current_layer.Z)
    
    # General case
    next_layer = model.layers[i + 1]
    return calculate_dZ(next_layer.W,
                        next_layer.dZ,
                        current_layer.Z,
                        current_layer.activation)
