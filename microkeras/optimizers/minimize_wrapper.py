from .minimize import minimize
from microkeras.operations.metrics.get_accuracy import get_accuracy
from microkeras.losses.categorical_crossentropy import categorical_crossentropy

def minimize_wrapper(self, model, X_train, Y_train, loss, num_iterations, batch_size=32):
    minimize(self, model, X_train, Y_train, loss, num_iterations, batch_size)
    
    # Calculate the final accuracy and loss
    final_acc = get_accuracy(model, X_train, Y_train)
    final_loss = categorical_crossentropy(Y_train, model.layers[-1].A)
    
    return final_acc, final_loss
