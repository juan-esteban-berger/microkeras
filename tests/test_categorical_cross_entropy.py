import pytest
import numpy as np
from microkeras.categorical_crossentropy import categorical_crossentropy

def test_categorical_crossentropy():
    print()
    print("Categorical Cross-Entropy function test:")
    # Create small 2D arrays for true labels and predicted probabilities
    Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Y_hat = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    print("True labels (Y):")
    print(Y)
    print("\nPredicted probabilities (Y_hat):")
    print(Y_hat)
    
    # Compute categoricall cross-entropy manually
    expected = -np.sum(Y * np.log(Y_hat + 1e-8))
    print("\nExpected output:")
    print(expected)
    
    # Compute categoricall cross-entropy using the function
    result = categorical_crossentropy(Y, Y_hat)
    print("\nActual output:")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    
    # Test with perfect predictions
    Y_perfect = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Y_hat_perfect = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    perfect_loss = categorical_crossentropy(Y_perfect, Y_hat_perfect)
    print("\nLoss with perfect predictions:")
    print(perfect_loss)
    
    # Assert that the loss for perfect predictions is very close to 0
    np.testing.assert_allclose(perfect_loss, 0, atol=1e-7)
    
    print("\nCategorical Cross-Entropy function test passed!")
