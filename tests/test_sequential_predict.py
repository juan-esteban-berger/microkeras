import pytest
import numpy as np
import os
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import mnist

def test_sequential_save_load():
    print("\nSequential save and load function test:")
    
    # Get the directory of the current file and set up model path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'test_models', 'test_model.json')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Load and preprocess MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    # Setup and train model
    model = Sequential([
        Dense(200, activation='sigmoid', input_shape=(784,)),
        Dense(200, activation='sigmoid'),
        Dense(10, activation='softmax')
    ])
    optimizer = SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train,
              y_train,
              batch_size=32,
              epochs=2)

    # Evaluate original model
    _, original_accuracy = model.evaluate(X_test, y_test)
    print(f"Original model accuracy: {original_accuracy:.4f}")

    # Save the model
    model.save(model_path)

    # Load the model
    loaded_model = model.load(model_path)

    # Compare weights and biases
    for original_layer, loaded_layer in zip(model.layers, loaded_model.layers):
        assert np.allclose(original_layer.W, loaded_layer.W), "Weights do not match"
        assert np.allclose(original_layer.b, loaded_layer.b), "Biases do not match"

    # Evaluate loaded model
    loaded_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    _, loaded_accuracy = loaded_model.evaluate(X_test, y_test)
    print(f"Loaded model accuracy: {loaded_accuracy:.4f}")

    # Assert that loaded model achieves reasonable accuracy
    assert loaded_accuracy > 0.8, f"Loaded model accuracy ({loaded_accuracy:.4f}) is below 80%"

    print("Sequential save and load function test passed!")

    # Clean up
    # os.remove(model_path)
