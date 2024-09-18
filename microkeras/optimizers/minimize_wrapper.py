from tqdm import tqdm
from microkeras.optimizers.minimize import minimize
from microkeras.losses.categorical_crossentropy import categorical_crossentropy
from microkeras.operations.metrics.get_accuracy import get_accuracy
from microkeras.operations.forward.forward import forward

def minimize_wrapper(optimizer, model, X_train, Y_train, loss, batch_size, epochs):
    history = {'accuracy': [], 'loss': []}
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        minimize(optimizer, model, X_train, Y_train, loss, batch_size)
        
        # Calculate accuracy and loss for the dataset after each epoch
        forward(model, X_train)  # Update the forward pass for entire dataset
        acc = get_accuracy(model, X_train, Y_train)
        loss_value = categorical_crossentropy(Y_train, model.layers[-1].A)
        
        history['accuracy'].append(acc)
        history['loss'].append(loss_value)
    
    return history
