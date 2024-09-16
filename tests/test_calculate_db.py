import pytest
import numpy as np
from microkeras.operations.backward.calculate_db import calculate_db

def test_calculate_db():
    print()
    print("Calculate db function test:")
    
    # Test parameters
    dZ = np.array([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]])
    m = 3  # number of examples
    
    print("Input parameters:")
    print("dZ:")
    print(dZ)
    print(f"m: {m}")
    
    # Calculate expected output manually
    expected_output = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    print("\nExpected output (manual calculation):")
    print(expected_output)
    
    # Calculate db using the function
    result = calculate_db(dZ, m)
    print("\nActual output (calculate_db function):")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected_output, rtol=1e-7, atol=1e-7)
    
    # Check if the output shape is correct
    expected_shape = (dZ.shape[0], 1)
    print(f"\nExpected shape: {expected_shape}, Actual shape: {result.shape}")
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"
    
    # Test with different dimensions
    print("\nTesting with different dimensions:")
    dZ_2 = np.random.randn(5, 10)
    m_2 = 10
    result_2 = calculate_db(dZ_2, m_2)
    expected_shape_2 = (dZ_2.shape[0], 1)
    print(f"Expected shape: {expected_shape_2}, Actual shape: {result_2.shape}")
    assert result_2.shape == expected_shape_2, f"Expected shape {expected_shape_2}, but got {result_2.shape}"
    
    print("\nCalculate db function test passed!")
