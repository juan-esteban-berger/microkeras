import pytest
import numpy as np
from microkeras.softmax import softmax

def test_softmax():
    print()
    print("Softmax test:")
    # Create a small 2D sequential array of data
    x = np.array([[1, 2, 3, 4, 5],
                  [0, 1, -1, 2, -2]])
    print("Input array:")
    print(x)
    
    # Compute softmax manually
    exp_shifted = np.exp(x - np.max(x))
    expected = exp_shifted / (exp_shifted.sum(axis=0) + 1e-8)
    print("\nExpected output:")
    print(expected)
    
    # Compute softmax using the
    result = softmax(x)
    print("\nActual output:")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    
    # Check that the sum of probabilities for each column is close to 1
    exp_row_sums = np.ones((1, 5))
    print("\nExpected sum of probabilities for each row:")
    print(exp_row_sums)
    row_sums = np.sum(result, axis=0).reshape(1, 5)
    print("\nSum of probabilities for each row:")
    print(row_sums)
    np.testing.assert_allclose(row_sums, exp_row_sums, rtol=1e-5, atol=1e-5)
    
    print("\nSoftmax test passed!")
