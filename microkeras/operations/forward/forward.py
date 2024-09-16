from .calculate_Z import calculate_Z
from .calculate_A import calculate_A
from .forward_layer import forward_layer

def forward(nn, X):
    A = X
    for layer in nn.layers:
        Z, A = forward_layer(layer, A)
        layer.Z = Z
        layer.A = A
    return A
