import pytest
import numpy as np
import os
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import mnist

@pytest.fixture(scope="module")
def mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    return X_train, y_train, X_test, y_test

@pytest.fixture(scope="module")
def model():
    return Sequential([
        Dense(200, activation='sigmoid', input_shape=(784,)),
        Dense(200, activation='sigmoid'),
        Dense(10, activation='softmax')
    ])

@pytest.fixture(scope="module")
def optimizer():
    return SGD(learning_rate=0.1)

def test_mnist_end_to_end(mnist_data, model, optimizer, tmp_path):
    X_train, y_train, X_test, y_test = mnist_data

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train,
                        y_train,
                        batch_size=32,
                        epochs=10)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    assert test_accuracy > 0.8, f"Test accuracy {test_accuracy:.4f} is below 80%"

    # Make predictions
    predictions = model.predict(X_test[:5])
    print("Predictions for the first 5 test samples:")
    print(np.argmax(predictions, axis=1))
    print("Actual labels:")
    print(np.argmax(y_test[:5], axis=1))

    # Save the model
    model_path = tmp_path / "mnist_model.json"
    model.save(str(model_path))

    # Load the model
    loaded_model = Sequential.load(str(model_path))

    # Compile the loaded model
    loaded_model.compile(optimizer=optimizer,
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

    # Evaluate the loaded model
    loaded_test_loss, loaded_test_accuracy = loaded_model.evaluate(X_test, y_test)
    print(f"Loaded model test accuracy: {loaded_test_accuracy:.4f}")
    assert abs(test_accuracy - loaded_test_accuracy) < 1e-6, "Loaded model accuracy differs from original model"

    # Check training history
    assert len(history['accuracy']) == 10, "Training history should have 10 epochs"
    assert history['accuracy'][-1] > history['accuracy'][0], "Model accuracy should improve during training"
    assert history['loss'][-1] < history['loss'][0], "Model loss should decrease during training"

    # Print training history
    print("\nTraining History:")
    print("Epoch\tAccuracy\tLoss")
    for epoch, (accuracy, loss) in enumerate(zip(history['accuracy'], history['loss']), 1):
        print(f"{epoch}\t{accuracy:.4f}\t{loss:.4f}")

if __name__ == "__main__":
    pytest.main([__file__])
