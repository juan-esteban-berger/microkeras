from tqdm import tqdm
from .select_batch import select_batches
from .gradient_descent import gradient_descent

def minimize(self, model, X_train, Y_train, loss, num_iterations, batch_size):
    """
    Perform mini-batch gradient descent.

    Args:
    self: The SGD optimizer instance.
    model (Sequential): The neural network model.
    X_train (numpy.ndarray): Input training data.
    Y_train (numpy.ndarray): True labels for training data.
    loss (str): The loss function to use.
    num_iterations (int): Number of iterations to perform.
    batch_size (int): Size of the mini-batch.

    Returns:
    None
    """
    for _ in tqdm(range(num_iterations), desc="SGD Progress"):
        X_batch, Y_batch = select_batches(X_train, Y_train, batch_size)
        gradient_descent(model, X_batch, Y_batch, loss, self.learning_rate)
