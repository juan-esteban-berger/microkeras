from .calculate_Z import calculate_Z
from .calculate_A import calculate_A
from .forward_layer import forward_layer

def forward(nn, X):
    """
    Perform forward propagation through the entire neural network.

    Parameters:
    nn (Sequential): The neural network model.
    X (numpy.ndarray): The input data.

    Returns:
    numpy.ndarray: The output of the last layer (final predictions).

    Side effects:
    - Updates the Z and A attributes of each layer in the network.
    """
    A = X
    for layer in nn.layers:
        Z, A = forward_layer(layer, A)
        layer.Z = Z
        layer.A = A
    return A
