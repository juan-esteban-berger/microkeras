def build(self):
    """
    Build all layers in the Sequential model.

    This method calls the build method of each layer in the model,
    initializing their weights and biases.
    """
    for layer in self.layers:
        layer.build(layer.input_shape)
