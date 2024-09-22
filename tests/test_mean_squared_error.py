import pytest
import numpy as np
from microkeras.losses import mean_squared_error

def test_mean_squared_error():
    print()
    print("Mean Squared Error function test:")
    # Create small 2D arrays for true values and predicted values
    Y = np.array([[1, 2, 3], [4, 5, 6]])
    Y_hat = np.array([[1.1, 2.2, 2.9], [3.8, 5.1, 6.3]])
    print("True values (Y):")
    print(Y)
    print("\nPredicted values (Y_hat):")
    print(Y_hat)
    
    # Compute mean squared error manually
    expected = np.mean((Y - Y_hat) ** 2)
    print("\nExpected output:")
    print(expected)
    
    # Compute mean squared error using the function
    result = mean_squared_error(Y, Y_hat)
    print("\nActual output:")
    print(result)
    
    # Assert that the result matches the expected output
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    
    # Test with perfect predictions
    Y_perfect = np.array([[1, 2, 3], [4, 5, 6]])
    Y_hat_perfect = np.array([[1, 2, 3], [4, 5, 6]])
    perfect_loss = mean_squared_error(Y_perfect, Y_hat_perfect)
    print("\nLoss with perfect predictions:")
    print(perfect_loss)
    
    # Assert that the loss for perfect predictions is 0
    np.testing.assert_allclose(perfect_loss, 0, atol=1e-7)
    
    print("\nMean Squared Error function test passed!")
