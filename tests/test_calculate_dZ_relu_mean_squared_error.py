import pytest
import numpy as np
from microkeras.operations.backward.calculate_dZ_relu_mean_squared_error import (
    calculate_dZ_relu_mean_squared_error
)

def test_calculate_dZ_relu_mean_squared_error():
    print()
    print("Calculate dZ for ReLU with mean squared error function test:")
    
    # Test parameters
    A = np.array([[0.3, 0.7, 0.0],
                  [0.2, 0.1, 0.7]])
    Y = np.array([[0.2, 0.8, 0.1],
                  [0.3, 0.0, 0.6]])
    Z = np.array([[0.5, 1.0, -0.2],
                  [0.1, -0.3, 0.8]])
    
    print("Input parameters:")
    print("A (ReLU output):")
    print(A)
    print("Y (true values):")
    print(Y)
    print("Z (pre-activation):")
    print(Z)
    
    # Calculate expected output manually
    m = A.shape[1]
    dA = 2 * (A - Y) / m
    expected_output = dA * (Z > 0)
    print("\nExpected output (manual calculation):")
    print(expected_output)
    
    # Calculate dZ using the function
    result = calculate_dZ_relu_mean_squared_error(A, Y, Z)
    print("\nActual output (calculate_dZ_relu_mean_squared_error function):")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected_output, rtol=1e-7, atol=1e-7)
    
    # Check if the output shape is correct
    print(f"\nExpected shape: {A.shape}, Actual shape: {result.shape}")
    assert result.shape == A.shape, f"Expected shape {A.shape}, but got {result.shape}"
    
    # Test with different dimensions
    print("\nTesting with different dimensions:")
    A_2 = np.random.rand(5, 10)
    Y_2 = np.random.rand(5, 10)
    Z_2 = np.random.randn(5, 10)
    result_2 = calculate_dZ_relu_mean_squared_error(A_2, Y_2, Z_2)
    print(f"Expected shape: {A_2.shape}, Actual shape: {result_2.shape}")
    assert result_2.shape == A_2.shape, f"Expected shape {A_2.shape}, but got {result_2.shape}"
    
    print("\nCalculate dZ for ReLU with mean squared error function test passed!")
