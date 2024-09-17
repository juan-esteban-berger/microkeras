import pytest
import numpy as np
from microkeras.datasets import mnist

def test_mnist_loader():
    print()
    print("MNIST data loader test:")

    # Load the data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Check shapes
    assert X_train.shape[1:] == (28, 28), "X_train should have shape (n_samples, 28, 28)"
    assert X_test.shape[1:] == (28, 28), "X_test should have shape (n_samples, 28, 28)"
    assert y_train.ndim == 1, "y_train should be 1-dimensional"
    assert y_test.ndim == 1, "y_test should be 1-dimensional"

    # Check data types
    assert X_train.dtype == np.float32, "X_train should be float32"
    assert X_test.dtype == np.float32, "X_test should be float32"
    assert y_train.dtype == np.uint8, "y_train should be uint8"
    assert y_test.dtype == np.uint8, "y_test should be uint8"

    # Check value ranges
    assert np.all(X_train >= 0) and np.all(X_train <= 1), "X_train values should be in range [0, 1]"
    assert np.all(X_test >= 0) and np.all(X_test <= 1), "X_test values should be in range [0, 1]"
    assert np.all(y_train >= 0) and np.all(y_train <= 9), "y_train values should be in range [0, 9]"
    assert np.all(y_test >= 0) and np.all(y_test <= 9), "y_test values should be in range [0, 9]"

    # Check for 10 unique labels
    assert len(np.unique(y_train)) == 10, "y_train should have 10 unique labels"
    assert len(np.unique(y_test)) == 10, "y_test should have 10 unique labels"

    print("MNIST data loader test passed!")
