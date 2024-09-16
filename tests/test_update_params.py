import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.operations.backward.update_params import update_params

def test_update_params():
    print()
    print("Update parameters function test:")
    
    # Create a simple neural network
    model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(3,)),
        Dense(2, activation='softmax')
    ])
    model.build()
    
    # Set learning rate
    learning_rate = 0.1
    
    # Store initial parameters
    initial_params = []
    for layer in model.layers:
        initial_params.append({
            'W': layer.W.copy(),
            'b': layer.b.copy()
        })
    
    # Set dummy gradients
    for layer in model.layers:
        layer.dW = np.random.randn(*layer.W.shape)
        layer.db = np.random.randn(*layer.b.shape)
    
    # Manually calculate expected updated parameters
    expected_params = []
    for layer, init_param in zip(model.layers, initial_params):
        expected_W = init_param['W'] - learning_rate * layer.dW
        expected_b = init_param['b'] - learning_rate * layer.db
        expected_params.append({
            'W': expected_W,
            'b': expected_b
        })
    
    # Print initial parameters and gradients
    for i, (layer, init_param) in enumerate(zip(model.layers, initial_params)):
        print(f"\nLayer {i+1} initial parameters and gradients:")
        print(f"Initial W:\n{init_param['W']}")
        print(f"Initial b:\n{init_param['b']}")
        print(f"dW:\n{layer.dW}")
        print(f"db:\n{layer.db}")
    
    # Update parameters using the function
    update_params(model, learning_rate)
    
    # Print and check updated parameters
    print("\nUpdated parameters:")
    for i, (layer, expected) in enumerate(zip(model.layers, expected_params)):
        print(f"\nLayer {i+1}:")
        print(f"Expected W:\n{expected['W']}")
        print(f"Actual W:\n{layer.W}")
        print(f"Expected b:\n{expected['b']}")
        print(f"Actual b:\n{layer.b}")
        
        np.testing.assert_allclose(layer.W, expected['W'], rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(layer.b, expected['b'], rtol=1e-7, atol=1e-7)
    
    print("\nUpdate parameters function test passed!")
