import pytest
import numpy as np
from microkeras.activations import relu

def test_relu():
    print()
    print("ReLU function test:")
    
    # Create a small 2D sequential array of data
    x = np.array([[-2, -1, 0, 1, 2],
                  [-1, -0.5, 0, 0.5, 1]])
    print("Input array:")
    print(x)
    
    # Compute ReLU manually
    expected = np.maximum(0, x)
    print("\nExpected output:")
    print(expected)
    
    # Compute ReLU using the function
    result = relu(x)
    print("\nActual output:")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    
    print("\nReLU function test passed!")
