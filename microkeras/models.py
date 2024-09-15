import numpy as np

class Sequential:
    def __init__(self, layers=None):
        self.layers = []

    def add(self, layer):
        # TODO:
        # Need to check if input_shape is none,
        # if it is none, make the input shape equal
        # to the output of the previous layer...
        self.layers.append(layer)

    def compile(self, optimizer, loss, metrics=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def fit(self, X, y, batch_size=32, epochs=1, verbose=1):
        history = {}
        return history

    def evaluate(self, X, y, verbose=0):
        loss, accuracy = None, None
        return loss, accuracy
