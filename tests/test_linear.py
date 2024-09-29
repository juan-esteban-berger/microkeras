import pytest
import numpy as np
from microkeras.losses import linear

def test_linear():
    print()
    print("Linear function test:")
    
    # Create a small 2D array of data
    x = np.array([[-2, -1, 0, 1, 2],
                  [-1, -0.5, 0, 0.5, 1]])
    print("Input array:")
    print(x)
    
    # Compute linear activation manually
    expected = x
    print("\nExpected output:")
    print(expected)
    
    # Compute linear activation using the function
    result = linear(x)
    print("\nActual output:")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    
    print("\nLinear function test passed!")
