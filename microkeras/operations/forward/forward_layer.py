from .calculate_Z import calculate_Z
from .calculate_A import calculate_A

def forward_layer(layer, A_prev):
    """
    Perform forward propagation for a single layer.

    Args:
        layer (Layer): The current layer object.
        A_prev (numpy.ndarray): Activation output from the previous layer.

    Returns:
        tuple: (Z, A)
            Z (numpy.ndarray): The linear combination output.
            A (numpy.ndarray): The activation output.

    Example:
        ```python
        layer = Dense(64, activation='relu')
        A_prev = np.random.randn(128, 32)
        Z, A = forward_layer(layer, A_prev)
        print(Z.shape, A.shape)
        ```
    """
    Z = calculate_Z(layer.W, A_prev, layer.b)
    A = calculate_A(Z, layer.activation)
    return Z, A
