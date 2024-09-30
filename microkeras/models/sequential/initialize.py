def initialize(self, layers):
    """
    Initialize the Sequential model with given layers.

    Parameters:
    layers (list): List of Layer instances to add to the model.
    """
    self.layers = []
    for layer in layers:
        self.add(layer)
    self.build()
