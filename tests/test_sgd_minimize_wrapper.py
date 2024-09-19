import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import mnist
from microkeras.optimizers.minimize_wrapper import minimize_wrapper

def test_sgd_minimize_wrapper():
    print()
    print("Minimize wrapper function test:")
    
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1).T
    Y_train = np.eye(10)[y_train].T
    
    # Setup model
    model = Sequential([
        Dense(200, activation='sigmoid', input_shape=(784,)),
        Dense(200, activation='sigmoid'),
        Dense(10, activation='softmax')
    ])
    
    # Setup optimizer
    optimizer = SGD(learning_rate=0.1)
    
    # Test with metrics=['accuracy']
    print("\nTesting minimize_wrapper with metrics=['accuracy']:")
    history_accuracy = minimize_wrapper(optimizer,
                                        model,
                                        X_train,
                                        Y_train,
                                        'categorical_crossentropy',
                                        32,
                                        2,
                                        ['accuracy'])
    
    # Assertions for accuracy metric
    assert 'accuracy' in history_accuracy, "Accuracy should be in history"
    assert 'loss' in history_accuracy, "Loss should always be in history"
    assert len(history_accuracy['accuracy']) == 2, "There should be two accuracy entries"
    assert len(history_accuracy['loss']) == 2, "There should be two loss entries"
    assert history_accuracy['accuracy'][0] < history_accuracy['accuracy'][-1], "Accuracy should improve"
    assert history_accuracy['loss'][0] > history_accuracy['loss'][-1], "Loss should decrease"
    print("Test passed: Model trained correctly with accuracy metric")

    # Reset model
    model = Sequential([
        Dense(200, activation='sigmoid', input_shape=(784,)),
        Dense(200, activation='sigmoid'),
        Dense(10, activation='softmax')
    ])

    # Test with empty metrics list
    print("\nTesting minimize_wrapper with empty metrics list:")
    history_empty = minimize_wrapper(optimizer,
                                     model,
                                     X_train,
                                     Y_train,
                                     'categorical_crossentropy',
                                     32,
                                     2,
                                     [])
    
    # Assertions for empty metrics list
    assert 'accuracy' not in history_empty, "Accuracy should not be in history"
    assert 'loss' in history_empty, "Loss should always be in history"
    assert len(history_empty['loss']) == 2, "There should be two loss entries"
    assert history_empty['loss'][0] > history_empty['loss'][-1], "Loss should decrease"
    print("Test passed: Model trained correctly with empty metrics list")

    print("\nMinimize wrapper function test passed!")
