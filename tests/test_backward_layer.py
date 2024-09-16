import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.activations import sigmoid, softmax
from microkeras.operations.forward.forward import forward
from microkeras.operations.backward.backward_layer import backward_layer

def test_backward_layer():
    print()
    print("Backward layer function test:")
    
    # Create a simple neural network
    model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(3,)),
        Dense(2, activation='softmax')
    ])
    model.build()
    
    # Set up test data
    X = np.array([[0.1, 0.4],
                  [0.2, 0.5],
                  [0.3, 0.6]])
    Y = np.array([[1, 0],
                  [0, 1]])
    m = X.shape[1]
    loss = 'categorical_crossentropy'
    
    print("Input (X):")
    print(X)
    print("Output (Y):")
    print(Y)
    
    # Perform forward propagation
    A = forward(model, X)
    
    # Test backward_layer for the output layer
    i = 1  # index of the output layer
    A_prev = model.layers[0].A
    dZ, dW, db = backward_layer(model, i, X, Y, A_prev, loss, m)
    
    print("\nOutput layer gradients:")
    print("dZ:")
    print(dZ)
    print("dW:")
    print(dW)
    print("db:")
    print(db)
    
    # Assert shapes
    assert dZ.shape == (model.layers[i].units, m), f"Incorrect shape for dZ"
    assert dW.shape == model.layers[i].W.shape, f"Incorrect shape for dW"
    assert db.shape == model.layers[i].b.shape, f"Incorrect shape for db"
    
    # Test backward_layer for the hidden layer
    i = 0  # index of the hidden layer
    A_prev = X
    dZ, dW, db = backward_layer(model, i, X, Y, A_prev, loss, m)
    
    print("\nHidden layer gradients:")
    print("dZ:")
    print(dZ)
    print("dW:")
    print(dW)
    print("db:")
    print(db)
    
    # Assert shapes
    assert dZ.shape == (model.layers[i].units, m), f"Incorrect shape for dZ"
    assert dW.shape == model.layers[i].W.shape, f"Incorrect shape for dW"
    assert db.shape == model.layers[i].b.shape, f"Incorrect shape for db"
    
    print("\nBackward layer function test passed!")
