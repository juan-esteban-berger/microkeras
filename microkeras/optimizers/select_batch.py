import numpy as np

def select_batches(X_train, Y_train, batch_size):
    """
    Randomly select a batch of samples from the training data.

    Parameters:
    X_train (numpy.ndarray): Input training data.
    Y_train (numpy.ndarray): True labels for training data.
    batch_size (int): The size of the batch to select.

    Returns:
    tuple: (X_batch, Y_batch)
    X_batch (numpy.ndarray): The selected batch of input data.
    Y_batch (numpy.ndarray): The corresponding batch of labels.

    This function handles both 1D and 2D label arrays (Y_train).
    """
    m = X_train.shape[1]  # number of training examples
    batch_indices = np.random.choice(m, batch_size, replace=False)
    X_batch = X_train[:, batch_indices]
    
    # Check if Y_train is 1D or 2D and select accordingly
    if Y_train.ndim == 1:
        Y_batch = Y_train[batch_indices]
    else:
        Y_batch = Y_train[:, batch_indices]
    
    return X_batch, Y_batch
