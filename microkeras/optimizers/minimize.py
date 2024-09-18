from tqdm import tqdm
from .select_batch import select_batches
from .gradient_descent import gradient_descent
from microkeras.operations.metrics.get_accuracy import get_accuracy
from microkeras.losses.categorical_crossentropy import categorical_crossentropy
from microkeras.operations.forward.forward import forward

def minimize(self, model, X_train, Y_train, loss, batch_size):
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

    accuracy_list = []
    loss_list = []
    for i in range(num_iterations):
        X_batch, Y_batch = select_batches(X_train, Y_train, batch_size)
        gradient_descent(model, X_batch, Y_batch, loss, self.learning_rate)

        # Update the progress bar at the specified frequency
        if i % update_frequency == 0 or i == num_iterations - 1:
            # Calculate accuracy and loss for the batch
            forward(model, X_batch)
            acc = get_accuracy(model, X_batch, Y_batch)
            loss_value = categorical_crossentropy(Y_batch, model.layers[-1].A)

            accuracy_list.append(acc)
            loss_list.append(loss_value)

            average_acc = sum(accuracy_list) / len(accuracy_list)
            average_loss = sum(loss_list) / len(loss_list)

            pbar.set_description(f"Batch {i+1}/{num_iterations} - Acc: {average_acc:.4f}, Loss: {average_loss:.4f}")
            pbar.update(update_frequency)

    pbar.close()
