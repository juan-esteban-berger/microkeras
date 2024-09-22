import pytest
import numpy as np
from microkeras.activations import relu_derivative

def test_relu_derivative():
    print()
    print("ReLU derivative function test:")
    
    # Create a small 2D sequential array of data
    x = np.array([[-2, -1, 0, 1, 2],
                  [-1, -0.5, 0, 0.5, 1]])
    print("Input array:")
    print(x)
    
    # Compute ReLU derivative manually
    expected = np.where(x > 0, 1.0, 0.0)
    print("\nExpected output:")
    print(expected)
    
    # Compute ReLU derivative using the function
    result = relu_derivative(x)
    print("\nActual output:")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    
    print("\nReLU derivative function test passed!")
