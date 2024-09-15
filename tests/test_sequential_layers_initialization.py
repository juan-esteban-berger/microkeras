import numpy as np
import pytest
from microkeras import Sequential
from microkeras.layers import Dense

def test_sequential_layers_initialization():
    print()
    print("Sequential layers initialization test:")

    # Expected values
    expected_units = [200, 200, 10]
    expected_activations = ['sigmoid', 'sigmoid', 'softmax']
    expected_input_shapes = [784, 200, 200]

    print("\nExpected model structure:")
    for i in range(3):
        print(f"Layer {i}: Dense(units={expected_units[i]}, activation='{expected_activations[i]}', input_shape={expected_input_shapes[i]})")

    print("\nInitializing layers...")
    model = Sequential([
        Dense(200, activation='sigmoid', input_shape=(784,)),
        Dense(200, activation='sigmoid'),
        Dense(10, activation='softmax')
    ])

    print("\nActual model structure:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: Dense(units={layer.units}, activation='{layer.activation}', input_shape={layer.input_shape})")
        print(f"  Weight shape: {layer.W.shape}")
        print(f"  Bias shape: {layer.b.shape}")

    print("\nTesting layer properties:")
    for i, layer in enumerate(model.layers):
        print(f"\nLayer {i}:")
        print(f"  Expected input shape: {expected_input_shapes[i]}, Actual: {layer.input_shape}")
        print(f"  Expected output shape (units): {expected_units[i]}, Actual: {layer.units}")
        print(f"  Expected activation: {expected_activations[i]}, Actual: {layer.activation}")
        print(f"  Weight shape: {layer.W.shape}")
        print(f"  Bias shape: {layer.b.shape}")

        # Test input shapes
        np.testing.assert_equal(layer.input_shape, expected_input_shapes[i])

        # Test output shapes (units)
        np.testing.assert_equal(layer.units, expected_units[i])

        # Test activations
        np.testing.assert_equal(layer.activation, expected_activations[i])

        # Test weight and bias initializations
        np.testing.assert_equal(layer.W.shape, (layer.units, layer.input_shape))
        np.testing.assert_equal(layer.b.shape, (layer.units, 1))

        # Check if W and b contain random values
        print(f"  Weights randomly initialized: {not np.allclose(layer.W, np.zeros_like(layer.W))}")
        print(f"  Biases randomly initialized: {not np.allclose(layer.b, np.zeros_like(layer.b))}")

        # Check if W and b are in the correct range (-0.5 to 0.5)
        print(f"  Weights in range [-0.5, 0.5]: {np.all(layer.W >= -0.5) and np.all(layer.W <= 0.5)}")
        print(f"  Biases in range [-0.5, 0.5]: {np.all(layer.b >= -0.5) and np.all(layer.b <= 0.5)}")

    print("\nAll layer properties are correct.")
    print("Sequential layers initialization test passed!")

if __name__ == "__main__":
    pytest.main([__file__])
