from microkeras.optimizers.minimize_wrapper import minimize_wrapper
import numpy as np

def fit(self, X_train, y_train, batch_size=32, epochs=1):
    """
    Train the model on the given data.

    Args:
        X_train (numpy.ndarray): Training input data.
        y_train (numpy.ndarray): True labels for the training data.
        batch_size (int): Number of samples per gradient update. Default is 32.
        epochs (int): Number of epochs to train the model. Default is 1.

    Returns:
        dict: Training history containing loss and metric values.

    Raises:
        ValueError: If the model hasn't been compiled.

    Example:
        ```python
        history = model.fit(X_train, y_train, batch_size=32, epochs=10)
        ```
    """
    if not hasattr(self, 'optimizer') or not hasattr(self, 'loss') or not hasattr(self, 'metrics'):
        raise ValueError("Model must be compiled before training. Use model.compile() first.")

    X_train = X_train.T
    y_train = y_train.T

    history = minimize_wrapper(self.optimizer,
                               self,
                               X_train,
                               y_train,
                               self.loss,
                               batch_size,
                               epochs,
                               self.metrics)

    return history
