import pytest
import numpy as np
from microkeras.activations import linear_derivative

def test_linear_derivative():
    print()
    print("Linear derivative function test:")
    
    # Create a small 2D array of data
    x = np.array([[-2, -1, 0, 1, 2],
                  [-1, -0.5, 0, 0.5, 1]])
    print("Input array:")
    print(x)
    
    # Compute linear derivative manually
    expected = np.ones_like(x)
    print("\nExpected output:")
    print(expected)
    
    # Compute linear derivative using the function
    result = linear_derivative(x)
    print("\nActual output:")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    
    print("\nLinear derivative function test passed!")
