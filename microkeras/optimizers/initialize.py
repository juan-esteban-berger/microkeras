def initialize(self, learning_rate):
    """
    Initialize the optimizer with a learning rate.

    Args:
        learning_rate (float): The learning rate to use for parameter updates.

    Example:
        ```python
        optimizer = SGD()
        optimizer.initialize(0.01)
        ```
    """
    self.learning_rate = learning_rate
