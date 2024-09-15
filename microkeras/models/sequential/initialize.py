def initialize(self, layers):
    self.layers = []
    for layer in layers:
        self.add(layer)
    self.build()
