from .minimize_wrapper import minimize_wrapper

class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    This class implements the SGD optimization algorithm.

    Attributes:
    learning_rate (float): The learning rate for parameter updates.

    Methods:
    minimize_wrapper: A method to perform training over multiple epochs.
    """
    def __init__(self, learning_rate=0.01):
        """
        Initialize the SGD optimizer.

        Parameters:
        learning_rate (float): The learning rate to use for parameter updates. Default is 0.01.
        """
        self.learning_rate = learning_rate
    
    minimize_wrapper = minimize_wrapper
