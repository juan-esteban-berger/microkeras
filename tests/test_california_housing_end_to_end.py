import pytest
import numpy as np
import os
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import california_housing

@pytest.fixture(scope="module")
def california_housing_data():
    (X_train, y_train), (X_test, y_test) = california_housing.load_data()
    # Ensure y is 2D for the model
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return X_train, y_train, X_test, y_test

@pytest.fixture(scope="module")
def model():
    return Sequential([
        Dense(64, activation='relu', input_shape=(8,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

@pytest.fixture(scope="module")
def optimizer():
    return SGD(learning_rate=0.01)

def test_california_housing_end_to_end(california_housing_data, model, optimizer, tmp_path):
    X_train, y_train, X_test, y_test = california_housing_data

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=[])  # Empty list for metrics

    # Train the model
    history = model.fit(X_train,
                        y_train,
                        batch_size=16,
                        epochs=50)

    # Evaluate the model
    test_mse = model.evaluate(X_test, y_test)
    print(f"Test MSE: {test_mse:.4f}")
    assert test_mse < 1.0, f"Test MSE {test_mse:.4f} is above 1.0"

    # Make predictions
    predictions = model.predict(X_test[:5])
    print("Predictions for the first 5 test samples:")
    print(predictions.flatten())
    print("Actual values:")
    print(y_test[:5].flatten())

    # Save the model
    model_path = tmp_path / "california_housing_model.json"
    model.save(str(model_path))

    # Load the model
    loaded_model = Sequential.load(str(model_path))

    # Compile the loaded model
    loaded_model.compile(optimizer=optimizer,
                         loss='mean_squared_error',
                         metrics=[])

    # Evaluate the loaded model
    loaded_test_mse = loaded_model.evaluate(X_test, y_test)
    print(f"Loaded model test MSE: {loaded_test_mse:.4f}")
    assert abs(test_mse - loaded_test_mse) < 1e-6, "Loaded model MSE differs from original model"

    # Check training history
    assert len(history['loss']) == 50, "Training history should have 50 epochs"
    assert history['loss'][-1] < history['loss'][0], "Model loss should decrease during training"

    # Print training history
    print("\nTraining History:")
    print("Epoch\tLoss")
    for epoch, loss in enumerate(history['loss'], 1):
        print(f"{epoch}\t{loss:.4f}")

if __name__ == "__main__":
    pytest.main([__file__])
