def copy(self):
    new_model = self.__class__([])
    for layer in self.layers:
        new_layer = layer.copy()
        new_model.add(new_layer)
    
    return new_model
