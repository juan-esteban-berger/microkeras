def add(self, layer):
    """
    Add a layer to the Sequential model.

    If the layer doesn't have an input_shape and there are existing layers,
    the input shape is inferred from the previous layer's units.

    Args:
        layer (Layer): The layer to be added to the model.

    Example:
        ```python
        model = Sequential([])
        model.add(Dense(64, activation='relu', input_shape=(784,)))
        model.add(Dense(10, activation='softmax'))
        ```
    """
    if self.layers and layer.input_shape is None:
        prev_layer = self.layers[-1]
        layer.input_shape = prev_layer.units
    self.layers.append(layer)
