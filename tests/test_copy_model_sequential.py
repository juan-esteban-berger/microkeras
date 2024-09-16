import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense

def test_sequential_copy():
    print()
    print("Sequential model copy test:")
    
    # Create a simple neural network
    model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(3,)),
        Dense(2, activation='softmax')
    ])
    model.build()
    
    # Create a copy of the model
    model_copy = model.copy()
    
    print("\nOriginal model structure:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i+1}: {layer.units} units, {layer.activation} activation")
    
    print("\nCopied model structure:")
    for i, layer in enumerate(model_copy.layers):
        print(f"Layer {i+1}: {layer.units} units, {layer.activation} activation")
    
    # Check if the structures are the same
    assert len(model.layers) == len(model_copy.layers), "Number of layers doesn't match"
    for original_layer, copied_layer in zip(model.layers, model_copy.layers):
        assert original_layer.units == copied_layer.units, "Number of units doesn't match"
        assert original_layer.activation == copied_layer.activation, "Activation function doesn't match"
    
    # Check if weights and biases are the same but not sharing memory
    for original_layer, copied_layer in zip(model.layers, model_copy.layers):
        assert np.array_equal(original_layer.W, copied_layer.W), "Weights are not equal"
        assert np.array_equal(original_layer.b, copied_layer.b), "Biases are not equal"
        assert id(original_layer.W) != id(copied_layer.W), "Weights are sharing memory"
        assert id(original_layer.b) != id(copied_layer.b), "Biases are sharing memory"
    
    # Modify the original model
    model.layers[0].W[0, 0] = 100
    
    print("\nAfter modifying original model:")
    print(f"Original model first layer, first weight: {model.layers[0].W[0, 0]}")
    print(f"Copied model first layer, first weight: {model_copy.layers[0].W[0, 0]}")
    
    assert model.layers[0].W[0, 0] != model_copy.layers[0].W[0, 0], "Modifying original affected the copy"
    
    print("\nSequential model copy test passed!")
