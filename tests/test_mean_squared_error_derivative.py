import pytest
import numpy as np
from microkeras.losses import mean_squared_error_derivative

def test_mean_squared_error_derivative():
    print()
    print("Mean Squared Error Derivative function test:")
    # Create small 2D arrays for true values and predicted values
    Y = np.array([[1, 2, 3], [4, 5, 6]])
    Y_hat = np.array([[1.1, 2.2, 2.9], [3.8, 5.1, 6.3]])
    print("True values (Y):")
    print(Y)
    print("\nPredicted values (Y_hat):")
    print(Y_hat)
    
    # Compute mean squared error derivative manually
    m = Y.shape[1]  # number of samples
    expected = 2 * (Y_hat - Y) / m
    print("\nExpected output:")
    print(expected)
    
    # Compute mean squared error derivative using the function
    result = mean_squared_error_derivative(Y, Y_hat)
    print("\nActual output:")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    
    # Test with perfect predictions
    Y_perfect = np.array([[1, 2, 3], [4, 5, 6]])
    Y_hat_perfect = np.array([[1, 2, 3], [4, 5, 6]])
    perfect_derivative = mean_squared_error_derivative(Y_perfect, Y_hat_perfect)
    print("\nDerivative with perfect predictions:")
    print(perfect_derivative)
    
    # Assert that the derivative for perfect predictions is 0
    np.testing.assert_allclose(perfect_derivative, 0, atol=1e-7)
    
    print("\nMean Squared Error Derivative function test passed!")
