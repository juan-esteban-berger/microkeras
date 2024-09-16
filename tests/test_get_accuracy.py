import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.operations.forward.forward import forward
from microkeras.operations.metrics.get_accuracy import get_accuracy

def test_get_accuracy():
    print()
    print("Get accuracy function test:")
    
    # Create a simple neural network
    model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(3,)),
        Dense(3, activation='softmax')
    ])
    model.build()
    
    # Create sample input data
    X = np.array([[0.1, 0.2, 0.3, 0.4],
                  [0.5, 0.6, 0.7, 0.8],
                  [0.9, 0.1, 0.2, 0.3]])
    
    # Create sample true labels (one-hot encoded)
    Y = np.array([[1, 0, 0, 1],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]])
    
    print("Input data shape:", X.shape)
    print("True labels shape:", Y.shape)
    
    # Perform forward propagation
    _ = forward(model, X)
    
    # Set dummy output activations
    model.layers[-1].A = np.array([[0.7, 0.1, 0.2, 0.9],
                                   [0.2, 0.8, 0.3, 0.1],
                                   [0.1, 0.1, 0.5, 0.0]])
    
    print("\nModel predictions:")
    print(model.layers[-1].A)
    
    # Calculate accuracy manually
    predictions = np.argmax(model.layers[-1].A, axis=0)
    Y_decoded = np.argmax(Y, axis=0)
    expected_accuracy = np.sum(predictions == Y_decoded) / Y_decoded.size
    
    print("\nExpected accuracy (manual calculation):", expected_accuracy)
    
    # Calculate accuracy using the function
    calculated_accuracy = get_accuracy(model, X, Y)
    
    print("Calculated accuracy (get_accuracy function):", calculated_accuracy)
    
    # Assert that the calculated accuracy matches the expected accuracy
    np.testing.assert_allclose(calculated_accuracy, expected_accuracy, rtol=1e-7, atol=1e-7)
    
    # Test with perfect predictions
    model.layers[-1].A = Y
    perfect_accuracy = get_accuracy(model, X, Y)
    print("\nAccuracy with perfect predictions:", perfect_accuracy)
    assert perfect_accuracy == 1.0, "Accuracy should be 1.0 with perfect predictions"
    
    # Test with completely wrong predictions
    model.layers[-1].A = 1 - Y
    worst_accuracy = get_accuracy(model, X, Y)
    print("Accuracy with completely wrong predictions:", worst_accuracy)
    assert worst_accuracy == 0.0, "Accuracy should be 0.0 with completely wrong predictions"
    
    print("\nGet accuracy function test passed!")
