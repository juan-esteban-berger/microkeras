def add(self, layer):
    if self.layers and layer.input_shape is None:
        prev_layer = self.layers[-1]
        layer.input_shape = prev_layer.units
    self.layers.append(layer)
