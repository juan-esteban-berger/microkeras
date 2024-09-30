from tqdm import tqdm
from .select_batch import select_batches
from .gradient_descent import gradient_descent
from microkeras.operations.metrics.get_accuracy import get_accuracy
from microkeras.losses.categorical_crossentropy import categorical_crossentropy
from microkeras.losses.mean_squared_error import mean_squared_error
from microkeras.operations.forward.forward import forward

def minimize(self, model, X_train, Y_train, loss, batch_size, metrics):
    """
    Minimize the loss function using mini-batch gradient descent.

    Parameters:
    self: The optimizer instance.
    model (Sequential): The neural network model.
    X_train (numpy.ndarray): Input training data.
    Y_train (numpy.ndarray): True labels for training data.
    loss (str): The loss function to use.
    batch_size (int): The size of each mini-batch.
    metrics (list): List of metrics to compute during training.

    This function performs mini-batch gradient descent, updating the model parameters
    to minimize the loss function. It uses a progress bar to show training progress
    and computes specified metrics.
    """
    # Number of training examples
    m = X_train.shape[1]
    # Number of batches per epoch
    num_iterations = m // batch_size  # Number of batches per epoch
    if m % batch_size != 0:
        # Ensure all data is used
        num_iterations += 1

    update_frequency = max(num_iterations // 30, 1)

    # Create tqdm progress bar
    pbar = tqdm(total=num_iterations, desc="SGD Progress")

    loss_list = []
    accuracy_list = [] if 'accuracy' in metrics else None

    for i in range(num_iterations):
        X_batch, Y_batch = select_batches(X_train, Y_train, batch_size)
        gradient_descent(model, X_batch, Y_batch, loss, self.learning_rate)

        # Update the progress bar at the specified frequency
        if i % update_frequency == 0 or i == num_iterations - 1:
            # Calculate loss for the batch
            forward(model, X_batch)
            if loss == 'categorical_crossentropy':
                loss_value = categorical_crossentropy(Y_batch, model.layers[-1].A)
            elif loss == 'mean_squared_error':
                loss_value = mean_squared_error(Y_batch, model.layers[-1].A)
            
            loss_list.append(loss_value)

            # Calculate accuracy for the batch if it's in metrics
            if 'accuracy' in metrics:
                acc = get_accuracy(model, X_batch, Y_batch)
                accuracy_list.append(acc)

            average_loss = sum(loss_list) / len(loss_list)
            desc = f"Batch {i+1}/{num_iterations} - Loss: {average_loss:.4f}"

            if 'accuracy' in metrics:
                average_acc = sum(accuracy_list) / len(accuracy_list)
                desc += f", Acc: {average_acc:.4f}"

            pbar.set_description(desc)
            pbar.update(update_frequency)

    pbar.close()
