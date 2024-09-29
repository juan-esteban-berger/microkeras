import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import california_housing

def test_sequential_evaluate_only_loss():
    print()
    print("Sequential evaluate function test (only loss):")
    
    # Load California Housing data
    (X_train, y_train), (X_test, y_test) = california_housing.load_data()
    
    # Take a small subset of the data for faster testing
    X_train, y_train = X_train[:1000], y_train[:1000]
    X_test, y_test = X_test[:100], y_test[:100]
    
    # Setup model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(8,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    # Compile the model
    optimizer = SGD(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[])
    
    # Fit the model
    model.fit(X_train, y_train, batch_size=32, epochs=1)
    
    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test)
    
    # Assertions
    assert isinstance(test_loss, float), "Loss should be a float"
    assert test_loss > 0, "Loss should be positive"
    
    print(f"Test loss: {test_loss:.4f}")
    
    print("Sequential evaluate function test (only loss) passed!")
