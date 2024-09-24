from tqdm import tqdm
from microkeras.optimizers.minimize import minimize
from microkeras.losses.categorical_crossentropy import categorical_crossentropy
from microkeras.losses.mean_squared_error import mean_squared_error
from microkeras.operations.metrics.get_accuracy import get_accuracy
from microkeras.operations.forward.forward import forward

def minimize_wrapper(optimizer, model, X_train, Y_train, loss, batch_size, epochs, metrics):
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
