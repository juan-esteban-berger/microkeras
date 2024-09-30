import json
import numpy as np

def load(cls, filename):
    """
    Load a Sequential model from a file.

    Parameters:
    filename (str): Path to the file containing the saved model.

    Returns:
    Sequential: A new Sequential model instance loaded from the file.
    """
    from microkeras.layers import Dense

    with open(filename, 'r') as f:
        model_data = json.load(f)
    
    new_model = cls([])
    
    for layer_data in model_data['layers']:
        layer = Dense(
            units=layer_data['units'],
            activation=layer_data['activation'],
            input_shape=layer_data['input_shape']
        )
        layer.W = np.array(layer_data['weights'])
        layer.b = np.array(layer_data['biases'])
        new_model.add(layer)
    
    return new_model
