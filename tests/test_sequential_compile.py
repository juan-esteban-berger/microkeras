import pytest
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD

def test_sequential_compile():
    print("\nCompile function test:")
    
    # Create a simple neural network
    model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(3,)),
        Dense(2, activation='softmax')
    ])
    
    # Compile the model with specific configurations
    optimizer = SGD(learning_rate=0.1)
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # Assertions to check that the compile function has set the attributes correctly
    assert model.optimizer == optimizer, "Optimizer not set correctly"
    assert model.loss == loss, "Loss not set correctly"
    assert model.metrics == metrics, "Metrics not set correctly"
    
    print("Test passed: Model compiled correctly with specified configurations.")
