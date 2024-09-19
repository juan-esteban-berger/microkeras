import numpy as np
from microkeras.operations.forward.forward import forward

def predict(self, X):
    # Transpose the input data
    X = X.T
    
    # Forward pass
    forward(self, X)
    
    # Return the output of the last layer
    predictions = self.layers[-1].A
    
    # Transpose the predictions to match the expected output shape
    return predictions.T
