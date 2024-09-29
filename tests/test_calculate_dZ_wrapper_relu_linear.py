import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.activations import relu, linear
from microkeras.operations.backward.calculate_dZ_wrapper import calculate_dZ_wrapper

def test_calculate_dZ_wrapper_relu_linear():
    print()
    print("Calculate dZ wrapper function test for ReLU and Linear activations:")
    
    # Create a simple neural network
    model = Sequential([
        Dense(4, activation='relu', input_shape=(3,)),
        Dense(3, activation='linear'),
        Dense(2, activation='linear')
    ])
    model.build()
    
    # Set up test data
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    Y = np.array([[0.7, 0.8], [0.9, 1.0]])
    
    # Perform forward pass
    A = X
    for layer in model.layers:
        Z = np.dot(layer.W, A) + layer.b
        if layer.activation == 'relu':
            A = relu(Z)
        elif layer.activation == 'linear':
            A = linear(Z)
        layer.Z = Z
        layer.A = A
    
    print("Test case 1: Last layer (linear) with mean squared error")
    dZ_last = calculate_dZ_wrapper(model, 2, Y, 'mean_squared_error')
    expected_dZ_last = 2 * (model.layers[-1].A - Y) / Y.shape[1]
    print("Expected dZ for last layer:")
    print(expected_dZ_last)
    print("Calculated dZ for last layer:")
    print(dZ_last)
    np.testing.assert_allclose(dZ_last, expected_dZ_last, rtol=1e-7, atol=1e-7)
    
    print("\nTest case 2: Hidden linear layer")
    model.layers[-1].dZ = dZ_last  # Set dZ for the last layer
    dZ_hidden_linear = calculate_dZ_wrapper(model, 1, Y, 'mean_squared_error')
    expected_dZ_hidden_linear = np.dot(model.layers[2].W.T, dZ_last)
    print("Expected dZ for hidden linear layer:")
    print(expected_dZ_hidden_linear)
    print("Calculated dZ for hidden linear layer:")
    print(dZ_hidden_linear)
    np.testing.assert_allclose(dZ_hidden_linear, expected_dZ_hidden_linear, rtol=1e-7, atol=1e-7)
    
    print("\nTest case 3: Hidden ReLU layer")
    model.layers[1].dZ = dZ_hidden_linear  # Set dZ for the linear layer
    dZ_hidden_relu = calculate_dZ_wrapper(model, 0, Y, 'mean_squared_error')
    expected_dZ_hidden_relu = np.dot(model.layers[1].W.T, dZ_hidden_linear) * (model.layers[0].Z > 0)
    print("Expected dZ for hidden ReLU layer:")
    print(expected_dZ_hidden_relu)
    print("Calculated dZ for hidden ReLU layer:")
    print(dZ_hidden_relu)
    np.testing.assert_allclose(dZ_hidden_relu, expected_dZ_hidden_relu, rtol=1e-7, atol=1e-7)
    
    print("\nCalculate dZ wrapper function test for ReLU and Linear activations passed!")
