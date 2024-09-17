from microkeras.operations.forward import forward
from microkeras.operations.backward import backward
from microkeras.operations.backward.update_params import update_params
from microkeras.operations.metrics.get_accuracy import get_accuracy
from microkeras.losses.categorical_crossentropy import categorical_crossentropy

def gradient_descent(model, X_train, Y_train, loss, learning_rate):
    forward(model, X_train)
    backward(model, X_train, Y_train, loss)
    update_params(model, learning_rate)
    
    # Calculate accuracy
    acc = get_accuracy(model, X_train, Y_train)
    
    # Calculate loss
    if loss == 'categorical_crossentropy':
        loss_val = categorical_crossentropy(Y_train, model.layers[-1].A)
    
    return acc, loss_val
