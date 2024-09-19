def compile(model, optimizer, loss, metrics):
    model.optimizer = optimizer
    model.loss = loss
    model.metrics = metrics
