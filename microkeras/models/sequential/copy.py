def copy(self):
    """
    Create a deep copy of the Sequential model.

    Returns:
        Sequential: A new Sequential model instance with copied layers.

    Note:
        This method creates a completely independent copy of the model,
        including all layers and their parameters.
    """
    new_model = self.__class__([])
    for layer in self.layers:
        new_layer = layer.copy()
        new_model.add(new_layer)
    
    return new_model
