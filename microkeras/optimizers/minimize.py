from tqdm import tqdm
from .select_batch import select_batches
from .gradient_descent import gradient_descent

def minimize(self, model, X_train, Y_train, loss, num_iterations, batch_size):
    for _ in tqdm(range(num_iterations), desc="SGD Progress"):
        X_batch, Y_batch = select_batches(X_train, Y_train, batch_size)
        gradient_descent(model, X_batch, Y_batch, loss, self.learning_rate)
