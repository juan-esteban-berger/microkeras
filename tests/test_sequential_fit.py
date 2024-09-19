import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import mnist

def test_sequential_fit():
    print()
    print("Sequential fit function test:")
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = np.eye(10)[y_train]
    
    model = Sequential([
        Dense(200, activation='sigmoid', input_shape=(784,)),
        Dense(200, activation='sigmoid'),
        Dense(10, activation='softmax')
    ])
    
    optimizer = SGD(learning_rate=0.1)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(X_train,
                        y_train,
                        batch_size=32,
                        epochs=2)
    
    assert len(history['accuracy']) == 2, "There should be two accuracy entries."
    assert len(history['loss']) == 2, "There should be two loss entries."
    assert history['accuracy'][0] < history['accuracy'][-1], "Accuracy should improve."
    assert history['loss'][0] > history['loss'][-1], "Loss should decrease."
    
    print("Test passed: Model fitted correctly and accuracy improved.")
