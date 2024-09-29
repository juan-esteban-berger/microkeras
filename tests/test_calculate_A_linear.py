import pytest
import numpy as np
from microkeras.operations.forward import calculate_A
from microkeras.activations import linear

def test_calculate_A_linear():
    print()
    print("Calculate A function test (Linear):")
    # Create sample input data
    Z_linear = np.array([[-1, 0, 1], [2, -2, 3]])
    
    print("Input for Linear (Z_linear):")
    print(Z_linear)
    
    # Test Linear activation
    expected_linear = linear(Z_linear)
    result_linear = calculate_A(Z_linear, 'linear')
    print("\nExpected output (Linear):")
    print(expected_linear)
    print("\nActual output (Linear):")
    print(result_linear)
    
    # Assert that the Linear result matches the expected output
    np.testing.assert_allclose(result_linear, expected_linear, rtol=1e-7, atol=1e-7)
    
    print("\nCalculate A function test (Linear) passed!")
