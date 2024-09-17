import numpy as np
from tqdm import tqdm
from microkeras.optimizers.gradient_descent import gradient_descent
from microkeras.operations.metrics.get_accuracy import get_accuracy
from microkeras.losses.categorical_crossentropy import categorical_crossentropy

def minimize(self, model, X_train, Y_train, loss, num_iterations, batch_size=32):
    m = X_train.shape[1]  # number of training examples
    
    for _ in tqdm(range(num_iterations), desc="SGD Progress"):
        # Randomly select a mini-batch
        batch_indices = np.random.choice(m, batch_size, replace=False)
        X_batch = X_train[:, batch_indices]
        Y_batch = Y_train[:, batch_indices]
        
        # Perform gradient descent on the mini-batch
        gradient_descent(model, X_batch, Y_batch, loss, self.learning_rate)
    
    # Calculate the final accuracy and loss
    final_acc = get_accuracy(model, X_train, Y_train)
    final_loss = categorical_crossentropy(Y_train, model.layers[-1].A)
    
    return final_acc, final_loss
