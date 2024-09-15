def initialize(self, units, activation, input_shape):
    self.units = units
    self.activation = activation
    self.input_shape = input_shape
    self.batch_size = None
    self.W = None
    self.b = None
    self.Z = None
    self.A = None
    self.dZ = None
    self.dW = None
    self.db = None
