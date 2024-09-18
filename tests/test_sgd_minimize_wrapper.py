import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import mnist
from microkeras.optimizers.minimize_wrapper import minimize_wrapper

def test_minimize_wrapper():
    print()
    print("Minimize function test:")
    
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
    
    # Minimize
    history = minimize_wrapper(optimizer,
                               model,
                               X_train,
                               Y_train,
                               'categorical_crossentropy',
                               32,
                               5)
    
    # Assertions to check training history
    assert len(history['accuracy']) == 5, "There should be five accuracy entries."
    assert len(history['loss']) == 5, "There should be five loss entries."
    assert history['accuracy'][0] < history['accuracy'][-1], "Accuracy should improve."
    assert history['loss'][0] > history['loss'][-1], "Loss should decrease."
    print("Test passed: Accuracy improved and loss decreased over epochs.")
