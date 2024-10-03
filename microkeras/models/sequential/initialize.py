def initialize(self, layers):
    """
    Initialize the Sequential model with given layers.

    Args:
        layers (list): List of Layer instances to add to the model.

    Note:
        This method is called internally by the Sequential constructor.
    """
    self.layers = []
    for layer in layers:
        self.add(layer)
    self.build()
