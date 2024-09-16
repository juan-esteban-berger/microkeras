import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.activations import sigmoid, softmax
from microkeras.operations.backward.calculate_db_wrapper import calculate_db_wrapper

def test_calculate_db_wrapper():
    print()
    print("Calculate db wrapper function test:")
    
    # Create a simple neural network
    model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(3,)),
        Dense(2, activation='softmax')
    ])
    model.build()
    
    # Set up test data
    X = np.array([[0.1, 0.2], 
                  [0.3, 0.4], 
                  [0.5, 0.6]])
    Y = np.array([[1, 0], 
                  [0, 1]])
    m = X.shape[1]  # number of training examples
    
    # Perform forward pass
    A = X
    for layer in model.layers:
        Z = np.dot(layer.W, A) + layer.b
        if layer.activation == 'sigmoid':
            A = sigmoid(Z)
        elif layer.activation == 'softmax':
            A = softmax(Z)
        layer.Z = Z
        layer.A = A
    
    # Set dummy dZ values for testing
    model.layers[0].dZ = np.random.randn(4, 2)
    model.layers[1].dZ = np.random.randn(2, 2)
    
    for i in range(len(model.layers)):
        print(f"\nTest case: Layer {i+1}")
        db = calculate_db_wrapper(model, i, m)
        expected_db = (1/m) * np.sum(model.layers[i].dZ, axis=1, keepdims=True)
        print(f"Expected db for layer {i+1}:")
        print(expected_db)
        print(f"Calculated db for layer {i+1}:")
        print(db)
        np.testing.assert_allclose(db, expected_db, rtol=1e-7, atol=1e-7)
    
    print("\nCalculate db wrapper function test passed!")
