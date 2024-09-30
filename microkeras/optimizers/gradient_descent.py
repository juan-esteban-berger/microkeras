from microkeras.operations.forward import forward
from microkeras.operations.backward import backward
from microkeras.operations.backward.update_params import update_params
from microkeras.operations.metrics.get_accuracy import get_accuracy
from microkeras.losses.categorical_crossentropy import categorical_crossentropy
from microkeras.losses.mean_squared_error import mean_squared_error

def gradient_descent(model, X_train, Y_train, loss, learning_rate):
    """
    Perform one step of gradient descent on the model.

    Parameters:
    model (Sequential): The neural network model.
    X_train (numpy.ndarray): Input training data.
    Y_train (numpy.ndarray): True labels for training data.
    loss (str): The loss function to use.
    learning_rate (float): The learning rate for parameter updates.

    Returns:
    tuple: (accuracy, loss_value)
    accuracy (float): The model's accuracy on the training data.
    loss_value (float): The loss value for the current state of the model.

    This function performs forward propagation, backward propagation, and parameter updates.
    It also calculates and returns the accuracy and loss for the current state of the model.
    """
    forward(model, X_train)
    backward(model, X_train, Y_train, loss)
    update_params(model, learning_rate)
    
    # Calculate accuracy
    acc = get_accuracy(model, X_train, Y_train)
    
    # Calculate loss
    if loss == 'categorical_crossentropy':
        loss_val = categorical_crossentropy(Y_train, model.layers[-1].A)
    elif loss == 'mean_squared_error':
        loss_val = mean_squared_error(Y_train, model.layers[-1].A)
    
    return acc, loss_val
