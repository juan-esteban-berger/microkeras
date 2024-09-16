import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.operations.forward.forward_layer import forward_layer
from microkeras.operations.forward.forward import forward

def test_forward():
    print()
    print("Forward propagation function test:")
    
    # Create a simple neural network
    model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(3,)),
        Dense(2, activation='softmax')
    ])
    model.build()
    
    # Create sample input
    X = np.array([[0.1, 0.4],
                  [0.2, 0.5],
                  [0.3, 0.6]])
    print("Input (X):")
    print(X)
    
    # Manually perform forward propagation
    A = X
    for layer in model.layers:
        Z, A = forward_layer(layer, A)
    
    expected_output = A
    print("\nExpected output (manual calculation):")
    print(expected_output)
    
    # Perform forward propagation using the forward function
    result = forward(model, X)
    
    print("\nActual output (forward function):")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected_output, rtol=1e-7, atol=1e-7)
    
    # Check if Z and A are stored in each layer
    for i, layer in enumerate(model.layers):
        print(f"\nLayer {i+1}:")
        print(f"Z shape: {layer.Z.shape}")
        print(f"A shape: {layer.A.shape}")
        assert hasattr(layer, 'Z'), f"Layer {i+1} is missing Z attribute"
        assert hasattr(layer, 'A'), f"Layer {i+1} is missing A attribute"
    
    print("\nForward propagation function test passed!")
