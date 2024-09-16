from .calculate_Z import calculate_Z
from .calculate_A import calculate_A

def forward_layer(layer, A_prev):
    Z = calculate_Z(layer.W, A_prev, layer.b)
    A = calculate_A(Z, layer.activation)
    return Z, A
