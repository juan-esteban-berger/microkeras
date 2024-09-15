import pytest
import numpy as np
from microkeras.sigmoid_derivative import sigmoid_derivative

def test_sigmoid_derivative():
    print()
    print("Sigmoid derivative function test:")
    # Create a small 2D sequential array of data
    x = np.array([[-2, -1, 0, 1, 2],
                  [-1, -0.5, 0, 0.5, 1]])
    print("Input array:")
    print(x)
    
    # Compute sigmoid derivative manually
    activation = 1 / (1 + np.exp(-x))
    expected = activation * (1 - activation)
    print("\nExpected output:")
    print(expected)
    
    # Compute sigmoid derivative using the function
    result = sigmoid_derivative(x)
    print("\nActual output:")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    
    print("\nSigmoid derivative function test passed!")
