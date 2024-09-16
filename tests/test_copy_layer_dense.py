import pytest
import numpy as np
from microkeras.layers.dense import Dense

def test_copy_layer_dense():
    print()
    print("Dense layer test:")
    
    # Test layer initialization
    layer = Dense(4, activation='sigmoid', input_shape=(3,))
    print("\nInitialized Dense layer:")
    print(f"Units: {layer.units}")
    print(f"Activation: {layer.activation}")
    print(f"Input shape: {layer.input_shape}")
    
    assert layer.units == 4, "Units not set correctly"
    assert layer.activation == 'sigmoid', "Activation not set correctly"
    assert layer.input_shape == (3,), "Input shape not set correctly"
    
    # Test layer building
    layer.build(3)
    print("\nBuilt Dense layer:")
    print(f"Weight shape: {layer.W.shape}")
    print(f"Bias shape: {layer.b.shape}")
    
    assert layer.W.shape == (4, 3), "Weight shape is incorrect"
    assert layer.b.shape == (4, 1), "Bias shape is incorrect"
    
    # Test layer copying
    copied_layer = layer.copy()
    print("\nCopied Dense layer:")
    print(f"Original layer weights:\n{layer.W}")
    print(f"Copied layer weights:\n{copied_layer.W}")
    
    assert np.array_equal(layer.W, copied_layer.W), "Weights not copied correctly"
    assert np.array_equal(layer.b, copied_layer.b), "Biases not copied correctly"
    assert id(layer.W) != id(copied_layer.W), "Weight arrays share the same memory"
    assert id(layer.b) != id(copied_layer.b), "Bias arrays share the same memory"
    
    # Modify the original layer
    layer.W[0, 0] = 100
    print("\nAfter modifying original layer:")
    print(f"Original layer weights:\n{layer.W}")
    print(f"Copied layer weights:\n{copied_layer.W}")
    
    assert not np.array_equal(layer.W, copied_layer.W), "Modifying original affected the copy"
    
    print("\nDense layer test passed!")
