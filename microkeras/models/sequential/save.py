import json
import numpy as np

def save(self, filename):
    """
    Save the Sequential model to a file.

    Parameters:
    filename (str): Path where the model should be saved.
    """
    model_data = {
        'layers': []
    }
    for layer in self.layers:
        layer_data = {
            'type': 'Dense',
            'units': layer.units,
            'activation': layer.activation,
            'input_shape': layer.input_shape,
            'weights': layer.W.tolist(),
            'biases': layer.b.tolist()
        }
        model_data['layers'].append(layer_data)
    
    with open(filename, 'w') as f:
        json.dump(model_data, f)
