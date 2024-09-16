from microkeras.operations.backward.calculate_dZ import calculate_dZ
from microkeras.operations.backward.calculate_dZ_softmax_categorical_crossentropy import (
    calculate_dZ_softmax_categorical_crossentropy
)

def calculate_dZ_wrapper(model, i, Y, loss):
    current_layer = model.layers[i]
    
    if (i == len(model.layers) - 1 and
        current_layer.activation == 'softmax' and
        loss == 'categorical_crossentropy'):
        return calculate_dZ_softmax_categorical_crossentropy(current_layer.A, Y)
    
    # General case
    next_layer = model.layers[i + 1]
    return calculate_dZ(next_layer.W,
                        next_layer.dZ,
                        current_layer.Z,
                        current_layer.activation)
