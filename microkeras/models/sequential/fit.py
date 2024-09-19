from microkeras.optimizers.minimize_wrapper import minimize_wrapper
import numpy as np

def fit(self, X_train, y_train, batch_size=32, epochs=1):
    if not hasattr(self, 'optimizer') or not hasattr(self, 'loss'):
        raise ValueError("Model must be compiled before training. Use model.compile() first.")

    X_train = X_train.T
    y_train = y_train.T

    history = minimize_wrapper(self.optimizer,
                               self,
                               X_train,
                               y_train,
                               self.loss,
                               batch_size,
                               epochs)

    return history
