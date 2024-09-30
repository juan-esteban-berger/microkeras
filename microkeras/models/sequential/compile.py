def compile(model, optimizer, loss, metrics):
    """
    Compile the Sequential model.

    This method configures the model for training by setting the optimizer,
    loss function, and metrics.

    Parameters:
    model (Sequential): The model to compile.
    optimizer (Optimizer): The optimizer to use for training.
    loss (str): The loss function to use.
    metrics (list): List of metrics to be evaluated during training and testing.
    """
    model.optimizer = optimizer
    model.loss = loss
    model.metrics = metrics
