import pytest
import numpy as np
from microkeras.operations.backward.calculate_dW import calculate_dW

def test_calculate_dW():
    print()
    print("Calculate dW function test:")
    
    # Test parameters
    dZ = np.array([[0.1, 0.2],
                   [0.3, 0.4]])
    A_prev = np.array([[0.5, 0.6],
                       [0.7, 0.8],
                       [0.9, 1.0]])
    m = 2  # number of examples
    
    print("Input parameters:")
    print("dZ:")
    print(dZ)
    print("A_prev:")
    print(A_prev)
    print(f"m: {m}")
    
    # Calculate expected output manually
    expected_output = (1 / m) * np.dot(dZ, A_prev.T)
    print("\nExpected output (manual calculation):")
    print(expected_output)
    
    # Calculate dW using the function
    result = calculate_dW(dZ, A_prev, m)
    print("\nActual output (calculate_dW function):")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected_output, rtol=1e-7, atol=1e-7)
    
    # Check if the output is of type float64
    print(f"\nOutput dtype: {result.dtype}")
    assert result.dtype == np.float64, f"Expected dtype float64, but got {result.dtype}"
    
    # Test with different dimensions
    print("\nTesting with different dimensions:")
    dZ_2 = np.random.randn(5, 3)
    A_prev_2 = np.random.randn(10, 3)
    m_2 = 3
    result_2 = calculate_dW(dZ_2, A_prev_2, m_2)
    expected_shape = (dZ_2.shape[0], A_prev_2.shape[0])
    print(f"Expected shape: {expected_shape}, Actual shape: {result_2.shape}")
    assert result_2.shape == expected_shape, f"Expected shape {expected_shape}, but got {result_2.shape}"
    
    print("\nCalculate dW function test passed!")
