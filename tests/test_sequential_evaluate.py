import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import mnist

def test_sequential_evaluate():
    print()
    print("Sequential evaluate function test:")
    
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Preprocess the data
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    
    # Take a small subset of the data for faster testing
    X_train, y_train = X_train[:1000], y_train[:1000]
    X_test, y_test = X_test[:100], y_test[:100]
    
    # Setup model
    model = Sequential([
        Dense(200, activation='sigmoid', input_shape=(784,)),
        Dense(200, activation='sigmoid'),
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    optimizer = SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fit the model
    model.fit(X_train, y_train, batch_size=32, epochs=1)
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    # Assertions
    assert isinstance(test_loss, float), "Loss should be a float"
    assert isinstance(test_accuracy, float), "Accuracy should be a float"
    assert 0 <= test_accuracy <= 1, "Accuracy should be between 0 and 1"
    
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    print("Sequential evaluate function test passed!")
