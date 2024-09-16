import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.activations import sigmoid, softmax
from microkeras.operations.forward.forward import forward
from microkeras.operations.backward.backward import backward
from microkeras.operations.backward.backward_layer import backward_layer

def test_backward():
    print()
    print("Backward propagation function test:")
    
    # Create a simple neural network
    model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(3,)),
        Dense(2, activation='softmax')
    ])
    model.build()
    
    # Create sample input and output
    X = np.array([[0.1, 0.4],
                  [0.2, 0.5],
                  [0.3, 0.6]])
    Y = np.array([[1, 0],
                  [0, 1]])
    print("Input (X):")
    print(X)
    print("Output (Y):")
    print(Y)
    
    # Perform forward propagation
    A = forward(model, X)
    
    # Manually perform backward propagation
    m = X.shape[1]
    A2 = model.layers[1].A
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, model.layers[0].A.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(model.layers[1].W.T, dZ2) * (model.layers[0].A * (1 - model.layers[0].A))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    expected_gradients = [
        {'dZ': dZ1, 'dW': dW1, 'db': db1},
        {'dZ': dZ2, 'dW': dW2, 'db': db2}
    ]
    
    print("\nExpected gradients (manual calculation):")
    for i, grad in enumerate(expected_gradients):
        print(f"Layer {i+1}:")
        print(f"dZ:\n{grad['dZ']}")
        print(f"dW:\n{grad['dW']}")
        print(f"db:\n{grad['db']}")
    
    # Perform backward propagation using the backward function
    backward(model, X, Y, 'categorical_crossentropy')
    
    print("\nActual gradients (backward function):")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i+1}:")
        print(f"dZ:\n{layer.dZ}")
        print(f"dW:\n{layer.dW}")
        print(f"db:\n{layer.db}")
    
    # Assert that the gradients match the expected output
    for i, layer in enumerate(model.layers):
        np.testing.assert_allclose(layer.dZ, expected_gradients[i]['dZ'], rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(layer.dW, expected_gradients[i]['dW'], rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(layer.db, expected_gradients[i]['db'], rtol=1e-7, atol=1e-7)
    
    # Check if gradients are stored in each layer
    for i, layer in enumerate(model.layers):
        print(f"\nLayer {i+1}:")
        print(f"dZ shape: {layer.dZ.shape}")
        print(f"dW shape: {layer.dW.shape}")
        print(f"db shape: {layer.db.shape}")
        assert hasattr(layer, 'dZ'), f"Layer {i+1} is missing dZ attribute"
        assert hasattr(layer, 'dW'), f"Layer {i+1} is missing dW attribute"
        assert hasattr(layer, 'db'), f"Layer {i+1} is missing db attribute"
    
    print("\nBackward propagation function test passed!")
