from tqdm import tqdm
from microkeras.optimizers.minimize import minimize
from microkeras.losses.categorical_crossentropy import categorical_crossentropy
from microkeras.losses.mean_squared_error import mean_squared_error
from microkeras.operations.metrics.get_accuracy import get_accuracy
from microkeras.operations.forward.forward import forward

def minimize_wrapper(optimizer, model, X_train, Y_train, loss, batch_size, epochs, metrics):
    """
    A wrapper function to perform multiple epochs of training using the minimize function.

    Parameters:
    optimizer: The optimizer instance.
    model (Sequential): The neural network model.
    X_train (numpy.ndarray): Input training data.
    Y_train (numpy.ndarray): True labels for training data.
    loss (str): The loss function to use.
    batch_size (int): The size of each mini-batch.
    epochs (int): The number of epochs to train for.
    metrics (list): List of metrics to compute during training.

    Returns:
    dict: A history dictionary containing the loss and specified metrics for each epoch.

    This function manages the training process over multiple epochs, calling the minimize
    function for each epoch and collecting the training history.
    """
    history = {}
    
    for metric in metrics:
        if metric in ['accuracy']:
            history[metric] = []
    history['loss'] = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        minimize(optimizer, model, X_train, Y_train, loss, batch_size, metrics)
        
        # Calculate metrics for the dataset after each epoch
        forward(model, X_train)  # Update the forward pass for entire dataset
        
        if 'accuracy' in metrics:
            acc = get_accuracy(model, X_train, Y_train)
            history['accuracy'].append(acc)
        
        if loss == 'categorical_crossentropy':
            loss_value = categorical_crossentropy(Y_train, model.layers[-1].A)
        elif loss == 'mean_squared_error':
            loss_value = mean_squared_error(Y_train, model.layers[-1].A)
        
        history['loss'].append(loss_value)
    
    return history
