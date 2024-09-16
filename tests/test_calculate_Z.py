import pytest
import numpy as np
from microkeras.operations.forward import calculate_Z

def test_calculate_Z():
    print()
    print("Calculate Z function test:")
    # Create sample input data
    W = np.array([[1, 2, 3], [4, 5, 6]])
    A_prev = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    b = np.array([[0.1], [0.2]])
    
    print("Weight matrix (W):")
    print(W)
    print("\nPrevious activation (A_prev):")
    print(A_prev)
    print("\nBias vector (b):")
    print(b)
    
    # Calculate expected output manually
    expected = np.dot(W, A_prev) + b
    print("\nExpected output:")
    print(expected)
    
    # Calculate Z using the function
    result = calculate_Z(W, A_prev, b)
    print("\nActual output:")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    
    # Test with different dimensions
    W2 = np.array([[1, 2], [3, 4], [5, 6]])
    A_prev2 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    b2 = np.array([[0.1], [0.2], [0.3]])
    
    expected2 = np.dot(W2, A_prev2) + b2
    result2 = calculate_Z(W2, A_prev2, b2)
    np.testing.assert_allclose(result2, expected2, rtol=1e-7, atol=1e-7)
    
    print("\nCalculate Z function test passed!")
