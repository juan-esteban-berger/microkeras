import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import mnist

def test_sequential_predict():
    print()
    print("Sequential predict function test:")
    
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Preprocess the data
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    
    # Setup model
    model = Sequential([
        Dense(200, activation='sigmoid', input_shape=(784,)),
        Dense(200, activation='sigmoid'),
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    optimizer = SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Fit the model
    model.fit(X_train,
              y_train,
              batch_size=32,
              epochs=2)
    
    # Make predictions
    predictions = model.predict(X_test[:5])
    
    # Assertions
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
    assert predictions.shape == (5, 10), f"Expected shape (5, 10), but got {predictions.shape}"
    assert np.allclose(np.sum(predictions, axis=1), 1.0), "Each prediction row should sum to approximately 1"
    
    print("Predictions shape:", predictions.shape)
    print("Predictions type:", type(predictions))
    print("Predictions:")
    print(predictions)
    
    print("Sequential predict function test passed!")
