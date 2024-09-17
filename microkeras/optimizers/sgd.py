from .minimize_wrapper import minimize_wrapper

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    minimize_wrapper = minimize_wrapper
