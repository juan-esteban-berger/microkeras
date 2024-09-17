import pytest
import numpy as np
from microkeras.datasets import california_housing

def test_california_housing_loader():
    print()
    print("California Housing data loader test:")

    # Load the data
    (X_train, y_train), (X_test, y_test) = california_housing.load_data()

    # Check shapes
    assert X_train.ndim == 2, "X_train should be 2-dimensional"
    assert X_test.ndim == 2, "X_test should be 2-dimensional"
    assert y_train.ndim == 1, "y_train should be 1-dimensional"
    assert y_test.ndim == 1, "y_test should be 1-dimensional"
    assert X_train.shape[1] == 8, "X_train should have 8 features"
    assert X_test.shape[1] == 8, "X_test should have 8 features"

    # Check data types
    assert X_train.dtype == np.float64, "X_train should be float64"
    assert X_test.dtype == np.float64, "X_test should be float64"
    assert y_train.dtype == np.float64, "y_train should be float64"
    assert y_test.dtype == np.float64, "y_test should be float64"

    # Check standardization of features
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-7), "X_train features should be centered around 0"
    assert np.allclose(X_train.std(axis=0), 1, atol=1e-7), "X_train features should have standard deviation of 1"

    # Check that y values are positive (housing prices)
    assert np.all(y_train > 0), "All y_train values should be positive"
    assert np.all(y_test > 0), "All y_test values should be positive"

    # Check that we have the expected number of samples (assuming 80-20 split)
    total_samples = len(y_train) + len(y_test)
    assert len(y_train) == int(0.8 * total_samples), "Training set should be 80% of the data"
    assert len(y_test) == int(0.2 * total_samples), "Test set should be 20% of the data"

    print("California Housing data loader test passed!")

if __name__ == "__main__":
    test_california_housing_loader()
