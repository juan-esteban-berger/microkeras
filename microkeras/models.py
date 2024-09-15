import numpy as np

class Sequential:
    def __init__(self, layers):
        self.layers = []
        for layer in layers:
            self.add(layer)
        self.build()

    def add(self, layer):
        if layer.input_shape is None:
            prev_layer = self.layers[-1]
            layer.input_shape = prev_layer.units
        self.layers.append(layer)

    def build(self):
        for layer in self.layers:
            layer.build(layer.input_shape)
