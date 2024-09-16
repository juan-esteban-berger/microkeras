import pytest
import numpy as np
from microkeras.operations.backward.calculate_dZ_softmax_categorical_crossentropy import (
    calculate_dZ_softmax_categorical_crossentropy
)

def test_calculate_dZ_softmax_categorical_crossentropy():
    print()
    print("Calculate dZ for softmax with categorical cross-entropy function test:")
    
    # Test parameters
    A = np.array([[0.3, 0.7, 0.0],
                  [0.2, 0.1, 0.7]])
    Y = np.array([[0, 1, 0],
                  [0, 0, 1]])
    
    print("Input parameters:")
    print("A (softmax probabilities):")
    print(A)
    print("Y (true labels, one-hot encoded):")
    print(Y)
    
    # Calculate expected output manually
    expected_output = A - Y
    print("\nExpected output (manual calculation):")
    print(expected_output)
    
    # Calculate dZ using the function
    result = calculate_dZ_softmax_categorical_crossentropy(A, Y)
    print("\nActual output (calculate_dZ_softmax_categorical_crossentropy function):")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected_output, rtol=1e-7, atol=1e-7)
    
    # Check if the output shape is correct
    print(f"\nExpected shape: {A.shape}, Actual shape: {result.shape}")
    assert result.shape == A.shape, f"Expected shape {A.shape}, but got {result.shape}"
    
    # Test with different dimensions
    print("\nTesting with different dimensions:")
    A_2 = np.random.rand(5, 10)
    Y_2 = np.eye(10)[np.random.choice(10, 5)]
    result_2 = calculate_dZ_softmax_categorical_crossentropy(A_2, Y_2)
    print(f"Expected shape: {A_2.shape}, Actual shape: {result_2.shape}")
    assert result_2.shape == A_2.shape, f"Expected shape {A_2.shape}, but got {result_2.shape}"
    
    print("\nCalculate dZ for softmax with categorical cross-entropy function test passed!")
