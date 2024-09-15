import pytest
import numpy as np
from microkeras.sigmoid import sigmoid

def test_sigmoid():
    print()
    print("Sigmoid function test:")
    
    # Create a small 2D sequential array of data
    x = np.array([[-2, -1, 0, 1, 2],
                  [-1, -0.5, 0, 0.5, 1]])
    print("Input array:")
    print(x)
    
    # Compute sigmoid manually
    expected = 1 / (1 + np.exp(-x))
    print("\nExpected output:")
    print(expected)
    
    # Compute sigmoid using the function
    result = sigmoid(x)
    print("\nActual output:")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    
    print("\nSigmoid function test passed!")
