import numpy as np

def select_batches(X_train, Y_train, batch_size):
    m = X_train.shape[1]  # number of training examples
    batch_indices = np.random.choice(m, batch_size, replace=False)
    X_batch = X_train[:, batch_indices]
    Y_batch = Y_train[:, batch_indices]
    return X_batch, Y_batch
