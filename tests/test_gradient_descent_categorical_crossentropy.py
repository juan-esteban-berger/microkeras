import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import gradient_descent
from microkeras.operations.forward import forward
from microkeras.operations.backward import backward
from microkeras.operations.backward.update_params import update_params
from microkeras.operations.metrics.get_accuracy import get_accuracy
from microkeras.losses.categorical_crossentropy import categorical_crossentropy

def test_gradient_descent_categorical_crossentropy():
    print()
    print("Gradient descent function test:")
    
    # Create a simple neural network
    original_model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(3,)),
        Dense(2, activation='softmax')
    ])
    original_model.build()
    
    # Create sample input data and labels
    X_train = np.array([[0.1, 0.2],
                        [0.3, 0.4],
                        [0.5, 0.6]])
    Y_train = np.array([[1, 0],
                        [0, 1]])
    
    print("Input data shape:", X_train.shape)
    print("Labels shape:", Y_train.shape)
    
    # Set hyperparameters
    learning_rate = 0.1
    loss = 'categorical_crossentropy'
    
    # Create a copy of the model for manual operations
    manual_model = original_model.copy()
    
    # Perform operations manually
    forward(manual_model, X_train)
    backward(manual_model, X_train, Y_train, loss)
    update_params(manual_model, learning_rate)
    
    manual_acc = get_accuracy(manual_model, X_train, Y_train)
    manual_loss = categorical_crossentropy(Y_train, manual_model.layers[-1].A)
    
    print("\nManual calculation results:")
    print(f"Accuracy: {manual_acc}")
    print(f"Loss: {manual_loss}")
    
    # Create another copy for gradient descent function
    gd_model = original_model.copy()
    
    # Perform gradient descent using the function
    gd_acc, gd_loss = gradient_descent(gd_model, X_train, Y_train, loss, learning_rate)
    
    print("\nGradient descent function results:")
    print(f"Accuracy: {gd_acc}")
    print(f"Loss: {gd_loss}")
    
    # Assert that the results are the same
    np.testing.assert_allclose(manual_acc, gd_acc, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(manual_loss, gd_loss, rtol=1e-7, atol=1e-7)
    
    # Check if the weights and biases are updated identically
    for manual_layer, gd_layer in zip(manual_model.layers, gd_model.layers):
        np.testing.assert_allclose(manual_layer.W, gd_layer.W, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(manual_layer.b, gd_layer.b, rtol=1e-7, atol=1e-7)
    
    print("\nGradient descent function test passed!")
