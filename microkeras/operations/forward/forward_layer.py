from .calculate_Z import calculate_Z
from .calculate_A import calculate_A

def forward_layer(layer, A_prev):
    """
    Perform forward propagation for a single layer.

    Parameters:
    layer (Layer): The current layer object.
    A_prev (numpy.ndarray): Activation output from the previous layer.

    Returns:
    tuple: (Z, A)
    Z (numpy.ndarray): The linear combination output.
    A (numpy.ndarray): The activation output.
    """
    Z = calculate_Z(layer.W, A_prev, layer.b)
    A = calculate_A(Z, layer.activation)
    return Z, A
