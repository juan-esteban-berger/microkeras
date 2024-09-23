import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.activations import sigmoid, softmax, relu
from microkeras.operations.backward.calculate_dZ_wrapper import calculate_dZ_wrapper

def test_calculate_dZ_wrapper():
    print()
    print("Calculate dZ wrapper function test:")
    
    # Create a simple neural network
    model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(3,)),
        Dense(3, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.build()
    
    # Set up test data
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    Y = np.array([[1, 0], [0, 1]])
    
    # Perform forward pass
    A = X
    for layer in model.layers:
        Z = np.dot(layer.W, A) + layer.b
        if layer.activation == 'sigmoid':
            A = sigmoid(Z)
        elif layer.activation == 'relu':
            A = relu(Z)
        elif layer.activation == 'softmax':
            A = softmax(Z)
        layer.Z = Z
        layer.A = A
    
    print("Test case 1: Last layer (softmax) with categorical crossentropy")
    dZ_last = calculate_dZ_wrapper(model, 2, Y, 'categorical_crossentropy')
    expected_dZ_last = model.layers[-1].A - Y
    print("Expected dZ for last layer:")
    print(expected_dZ_last)
    print("Calculated dZ for last layer:")
    print(dZ_last)
    np.testing.assert_allclose(dZ_last, expected_dZ_last, rtol=1e-7, atol=1e-7)
    
    print("\nTest case 2: Hidden ReLU layer")
    model.layers[-1].dZ = dZ_last  # Set dZ for the last layer
    dZ_hidden_relu = calculate_dZ_wrapper(model, 1, Y, 'categorical_crossentropy')
    expected_dZ_hidden_relu = np.dot(model.layers[2].W.T, dZ_last) * (model.layers[1].Z > 0)
    print("Expected dZ for hidden ReLU layer:")
    print(expected_dZ_hidden_relu)
    print("Calculated dZ for hidden ReLU layer:")
    print(dZ_hidden_relu)
    np.testing.assert_allclose(dZ_hidden_relu, expected_dZ_hidden_relu, rtol=1e-7, atol=1e-7)
    
    print("\nTest case 3: Hidden sigmoid layer")
    model.layers[1].dZ = dZ_hidden_relu  # Set dZ for the ReLU layer
    dZ_hidden_sigmoid = calculate_dZ_wrapper(model, 0, Y, 'categorical_crossentropy')
    expected_dZ_hidden_sigmoid = np.dot(model.layers[1].W.T, dZ_hidden_relu) * (model.layers[0].A * (1 - model.layers[0].A))
    print("Expected dZ for hidden sigmoid layer:")
    print(expected_dZ_hidden_sigmoid)
    print("Calculated dZ for hidden sigmoid layer:")
    print(dZ_hidden_sigmoid)
    np.testing.assert_allclose(dZ_hidden_sigmoid, expected_dZ_hidden_sigmoid, rtol=1e-7, atol=1e-7)
    
    print("\nCalculate dZ wrapper function test passed!")
