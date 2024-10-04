from .calculate_Z import calculate_Z
from .calculate_A import calculate_A
from .forward_layer import forward_layer

def forward(nn, X):
    """
    Perform forward propagation through the entire neural network.

    Args:
        nn (Sequential): The neural network model.
        X (numpy.ndarray): The input data.

    Returns:
        numpy.ndarray: The output of the last layer (final predictions).

    Example:
        ```python
        model = Sequential([
            Dense(64, activation='relu', input_shape=(128,)),
            Dense(10, activation='softmax')
        ])
        X = np.random.randn(128, 32)
        output = forward(model, X)
        print(output.shape)
        ```

    Note:
        This function updates the Z and A attributes of each layer in the network.
    """
    A = X
    for layer in nn.layers:
        Z, A = forward_layer(layer, A)
        layer.Z = Z
        layer.A = A
    return A
