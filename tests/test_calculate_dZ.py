import pytest
import numpy as np
from microkeras.operations.backward.calculate_dZ import calculate_dZ
from microkeras.activations import sigmoid_derivative

def test_calculate_dZ():
    print()
    print("Calculate dZ function test:")
    
    # Test parameters
    W_next = np.array([[0.1, 0.2],
                       [0.3, 0.4],
                       [0.5, 0.6]])
    dZ_next = np.array([[0.1],
                        [0.2],
                        [0.3]])
    Z = np.array([[0.5],
                  [0.6]])
    activation = 'sigmoid'
    
    print("Input parameters:")
    print("W_next:")
    print(W_next)
    print("dZ_next:")
    print(dZ_next)
    print("Z:")
    print(Z)
    print(f"Activation: {activation}")
    
    # Calculate expected output manually
    expected_output = np.dot(W_next.T, dZ_next) * sigmoid_derivative(Z)
    print("\nExpected output (manual calculation):")
    print(expected_output)
    
    # Calculate dZ using the function
    result = calculate_dZ(W_next, dZ_next, Z, activation)
    print("\nActual output (calculate_dZ function):")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected_output, rtol=1e-7, atol=1e-7)
    
    print("\nCalculate dZ function test passed!")
