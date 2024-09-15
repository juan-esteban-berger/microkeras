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

    def compile(self, optimizer, loss, metrics=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def fit(self, X, y, batch_size=32, epochs=1, verbose=1):
        # TODO:
        # Do not forget to transpose both X and Y
        # before running optimizer
        history = {}
        return history

    def evaluate(self, X, y, verbose=0):
        loss, accuracy = None, None
        return loss, accuracy
