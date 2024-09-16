import pytest
import numpy as np
from microkeras.layers import Dense
from microkeras.operations.forward.calculate_Z import calculate_Z
from microkeras.operations.forward.calculate_A import calculate_A
from microkeras.operations.forward.forward_layer import forward_layer

def test_forward_layer():
    print()
    print("Forward layer function test:")
    
    # Create a simple dense layer
    layer = Dense(4, activation='sigmoid', input_shape=(3,))
    layer.build(3)
    
    # Create sample input (3 features, 2 examples)
    A_prev = np.array([[0.1, 0.4],
                       [0.2, 0.5],
                       [0.3, 0.6]])
    print("Input (A_prev):")
    print(A_prev)
    
    # Manually perform forward propagation for the layer
    Z_expected = calculate_Z(layer.W, A_prev, layer.b)
    A_expected = calculate_A(Z_expected, layer.activation)
    
    print("\nExpected output (manual calculation):")
    print("Z_expected:")
    print(Z_expected)
    print("A_expected:")
    print(A_expected)
    
    # Perform forward propagation using the forward_layer function
    Z_result, A_result = forward_layer(layer, A_prev)
    
    print("\nActual output (forward_layer function):")
    print("Z_result:")
    print(Z_result)
    print("A_result:")
    print(A_result)
    
    # Assert that the results match the expected outputs
    np.testing.assert_allclose(Z_result, Z_expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(A_result, A_expected, rtol=1e-7, atol=1e-7)
    
    print("\nForward layer function test passed!")
