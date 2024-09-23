import warnings
warnings.filterwarnings("ignore", module="tqdm")
import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import california_housing
from microkeras.optimizers.minimize import minimize
from microkeras.losses.mean_squared_error import mean_squared_error

def test_sgd_minimize_mean_squared_error():
    print()
    print("Minimize function test with Mean Squared Error:")
    
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
    
    # Create SGD optimizer
    optimizer = SGD(learning_rate=learning_rate)
    
    # Test with MSE
    print("\nTesting minimize with Mean Squared Error:")
    initial_loss = mean_squared_error(y_train, model.predict(X_train.T).T)
    print(f"Initial MSE loss: {initial_loss:.4f}")
    
    minimize(optimizer, model, X_train, y_train, loss, batch_size, metrics=[])
    
    final_loss = mean_squared_error(y_train, model.predict(X_train.T).T)
    print(f"Final MSE loss: {final_loss:.4f}")
    
    assert final_loss < initial_loss, "Model MSE did not improve"
    assert final_loss < 10.0, f"Final MSE {final_loss:.4f} is above 10.0"
    
    # Reset model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(8,)),
        Dense(32, activation='relu'),
        Dense(1, activation='relu')
    ])
    
    # Test with different batch size
    print("\nTesting minimize with different batch size:")
    initial_loss = mean_squared_error(y_train, model.predict(X_train.T).T)
    print(f"Initial MSE loss: {initial_loss:.4f}")
    
    minimize(optimizer, model, X_train, y_train, loss, batch_size=64, metrics=[])
    
    final_loss = mean_squared_error(y_train, model.predict(X_train.T).T)
    print(f"Final MSE loss: {final_loss:.4f}")
    
    assert final_loss < initial_loss, "Model MSE did not improve with different batch size"
    assert final_loss < 10.0, f"Final MSE {final_loss:.4f} is above 10.0 with different batch size"
    
    print("\nMinimize function test with Mean Squared Error passed!")
