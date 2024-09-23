import pytest
import numpy as np
from microkeras.operations.forward import calculate_A
from microkeras.activations import relu

def test_calculate_A_relu():
    print()
    print("Calculate A function test (ReLU):")
    # Create sample input data
    Z_relu = np.array([[-1, 0, 1], [2, -2, 3]])
    
    print("Input for ReLU (Z_relu):")
    print(Z_relu)
    
    # Test ReLU activation
    expected_relu = relu(Z_relu)
    result_relu = calculate_A(Z_relu, 'relu')
    print("\nExpected output (ReLU):")
    print(expected_relu)
    print("\nActual output (ReLU):")
    print(result_relu)
    
    # Assert that the ReLU result matches the expected output
    np.testing.assert_allclose(result_relu, expected_relu, rtol=1e-7, atol=1e-7)
    
    print("\nCalculate A function test (ReLU) passed!")
