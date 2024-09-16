import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.activations import sigmoid, softmax
from microkeras.operations.backward.calculate_dW_wrapper import calculate_dW_wrapper

def test_calculate_dW_wrapper():
    print()
    print("Calculate dW wrapper function test:")
    
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
    
    print("Test case 1: First layer")
    dW_first = calculate_dW_wrapper(model, 0, X, m)
    expected_dW_first = (1/m) * np.dot(model.layers[0].dZ, X.T)
    print("Expected dW for first layer:")
    print(expected_dW_first)
    print("Calculated dW for first layer:")
    print(dW_first)
    np.testing.assert_allclose(dW_first, expected_dW_first, rtol=1e-7, atol=1e-7)
    
    print("\nTest case 2: Second layer")
    dW_second = calculate_dW_wrapper(model, 1, X, m)
    expected_dW_second = (1/m) * np.dot(model.layers[1].dZ, model.layers[0].A.T)
    print("Expected dW for second layer:")
    print(expected_dW_second)
    print("Calculated dW for second layer:")
    print(dW_second)
    np.testing.assert_allclose(dW_second, expected_dW_second, rtol=1e-7, atol=1e-7)
    
    print("\nCalculate dW wrapper function test passed!")
