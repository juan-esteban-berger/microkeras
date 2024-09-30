import numpy as np
from microkeras.operations.forward.forward import forward

def predict(self, X):
    """
    Generate output predictions for the input samples.

    Parameters:
    X (numpy.ndarray): Input data.

    Returns:
    numpy.ndarray: Predictions for the input data.
    """
    # Transpose the input data
    X = X.T
    
    # Forward pass
    forward(self, X)
    
    # Return the output of the last layer
    predictions = self.layers[-1].A
    
    # Transpose the predictions to match the expected output shape
    return predictions.T
