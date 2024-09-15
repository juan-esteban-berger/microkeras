def build(self):
    for layer in self.layers:
        layer.build(layer.input_shape)
