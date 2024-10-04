import numpy as np
from microkeras.operations.forward import forward

def get_accuracy(model, X, Y):
    """
    Calculate the accuracy of the model's predictions.

    Args:
        model (Sequential): The neural network model.
        X (numpy.ndarray): Input data, shape (n_features, n_samples).
        Y (numpy.ndarray): True labels in one-hot encoded format, shape (n_classes, n_samples).

    Returns:
        float: The accuracy of the model's predictions, as a value between 0 and 1.

    Example:
        ```python
        model = Sequential([
            Dense(64, activation='relu', input_shape=(784,)),
            Dense(10, activation='softmax')
        ])
        X_test = np.random.randn(784, 100)
        Y_test = np.eye(10)[np.random.randint(0, 10, 100)].T
        accuracy = get_accuracy(model, X_test, Y_test)
        print(f"Model accuracy: {accuracy}")
        ```

    Note:
        - This function assumes that the model's output and true labels are in one-hot encoded format.
        - The accuracy is calculated as the ratio of correct predictions to total predictions.
    """
    forward(model, X)
    predictions = np.argmax(model.layers[-1].A, axis=0)
    Y_decoded = np.argmax(Y, axis=0)
    return np.sum(predictions == Y_decoded) / Y_decoded.size
