from microkeras.operations.forward.forward import forward
from microkeras.losses.categorical_crossentropy import categorical_crossentropy
from microkeras.operations.metrics.get_accuracy import get_accuracy

def evaluate(self, X_test, y_test):
    if not hasattr(self, 'loss'):
        raise ValueError("Model must be compiled before evaluation. Use model.compile() first.")
    
    # Ensure input data is in the correct shape
    X_test = X_test.T
    y_test = y_test.T
    
    # Forward pass
    forward(self, X_test)
    
    # Calculate loss
    y_pred = self.layers[-1].A
    loss_value = categorical_crossentropy(y_test, y_pred)
    
    # Calculate accuracy
    acc = get_accuracy(self, X_test, y_test)
    
    # Return loss and accuracy as a tuple
    return (loss_value, acc)
