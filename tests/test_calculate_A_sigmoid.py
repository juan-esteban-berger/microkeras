import pytest
import numpy as np
from microkeras.calculate_A import calculate_A
from microkeras.sigmoid import sigmoid

def test_calculate_A_sigmoid():
    print()
    print("Calculate A function test (Sigmoid):")
    # Create sample input data
    Z_sigmoid = np.array([[-1, 0, 1], [2, -2, 3]])
    
    print("Input for Sigmoid (Z_sigmoid):")
    print(Z_sigmoid)
    
    # Test Sigmoid activation
    expected_sigmoid = sigmoid(Z_sigmoid)
    result_sigmoid = calculate_A(Z_sigmoid, 'sigmoid')
    print("\nExpected output (Sigmoid):")
    print(expected_sigmoid)
    print("\nActual output (Sigmoid):")
    print(result_sigmoid)
    
    # Assert that the Sigmoid result matches the expected output
    np.testing.assert_allclose(result_sigmoid, expected_sigmoid, rtol=1e-7, atol=1e-7)
    
    print("\nCalculate A function test (Sigmoid) passed!")
