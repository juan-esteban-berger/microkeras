def copy(self):
    """
    Create a deep copy of the Sequential model.

    Returns:
    Sequential: A new Sequential model instance with copied layers.
    """
    new_model = self.__class__([])
    for layer in self.layers:
        new_layer = layer.copy()
        new_model.add(new_layer)
    
    return new_model
