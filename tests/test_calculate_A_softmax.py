import pytest
import numpy as np
from microkeras.operations.forward import calculate_A
from microkeras.activations import softmax

def test_calculate_A_softmax():
    print()
    print("Calculate A function test (Softmax):")
    # Create sample input data
    Z_softmax = np.array([[1, 2, 3], [4, 5, 6]])
    
    print("Input for Softmax (Z_softmax):")
    print(Z_softmax)
    
    # Test Softmax activation
    expected_softmax = softmax(Z_softmax)
    result_softmax = calculate_A(Z_softmax, 'softmax')
    print("\nExpected output (Softmax):")
    print(expected_softmax)
    print("\nActual output (Softmax):")
    print(result_softmax)
    
    # Assert that the Softmax result matches the expected output
    np.testing.assert_allclose(result_softmax, expected_softmax, rtol=1e-7, atol=1e-7)
    
    print("\nCalculate A function test (Softmax) passed!")
