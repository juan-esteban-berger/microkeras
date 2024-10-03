def compile(model, optimizer, loss, metrics):
    """
    This method configures the model for training by setting the optimizer,
    loss function, and metrics.

    Args:
        model (Sequential): The model to compile.
        optimizer (Optimizer): The optimizer instance to use for training.
        loss (str): The name of the loss function to use (e.g., 'categorical_crossentropy').
        metrics (list): List of metric names to be evaluated during training and testing.

    Example:
        ```python
        model.compile(optimizer=SGD(learning_rate=0.01),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        ```
    """
    model.optimizer = optimizer
    model.loss = loss
    model.metrics = metrics
