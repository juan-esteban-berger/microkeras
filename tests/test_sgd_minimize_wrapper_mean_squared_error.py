import warnings
warnings.filterwarnings("ignore", module="tqdm")
import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import california_housing
from microkeras.optimizers.minimize_wrapper import minimize_wrapper
from microkeras.losses.mean_squared_error import mean_squared_error

def test_sgd_minimize_wrapper_mean_squared_error():
    print()
    print("Minimize wrapper function test with Mean Squared Error:")
    
    # Load California Housing data
    (X_train, y_train), (X_test, y_test) = california_housing.load_data()
    
    # Transpose the data to match the expected input shape
    X_train = X_train.T
    y_train = y_train.reshape(1, -1)
    
    print("Input data shape:", X_train.shape)
    print("Labels shape:", y_train.shape)
    
    # Create a simple neural network
    model = Sequential([
        Dense(64, activation='relu', input_shape=(8,)),
        Dense(32, activation='relu'),
        Dense(1, activation='relu')
    ])
    
    # Set hyperparameters
    learning_rate = 0.001
    loss = 'mean_squared_error'
    batch_size = 8
    epochs = 5
    
    # Create SGD optimizer
    optimizer = SGD(learning_rate=learning_rate)
    
    # Test minimize_wrapper
    print("\nTesting minimize_wrapper with Mean Squared Error:")
    initial_loss = mean_squared_error(y_train, model.predict(X_train.T).T)
    print(f"Initial MSE loss: {initial_loss:.4f}")
    
    history = minimize_wrapper(optimizer, model, X_train, y_train, loss, batch_size, epochs, metrics=[])
    
    final_loss = mean_squared_error(y_train, model.predict(X_train.T).T)
    print(f"Final MSE loss: {final_loss:.4f}")
    
    assert final_loss < initial_loss, "Model MSE did not improve"
    assert final_loss < 3.0, f"Final MSE {final_loss:.1f} is above 1.0"
    
    # Check if history contains loss values
    assert 'loss' in history, "History does not contain loss values"
    assert len(history['loss']) == epochs, f"Expected {epochs} loss values, but got {len(history['loss'])}"
    
    # Check if loss decreases over epochs
    assert history['loss'][0] > history['loss'][-1], "Loss did not decrease over epochs"
    
    print("\nLoss values over epochs:")
    for epoch, loss_value in enumerate(history['loss'], 1):
        print(f"Epoch {epoch}: {loss_value:.4f}")
    
    print("\nMinimize wrapper function test with Mean Squared Error passed!")
