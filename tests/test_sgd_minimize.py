import warnings
warnings.filterwarnings("ignore", module="tqdm")
import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.operations.metrics.get_accuracy import get_accuracy
from microkeras.datasets import mnist
from microkeras.optimizers.minimize import minimize
from microkeras.operations.forward.forward import forward

def test_sgd_minimize():
    print()
    print("Minimize function test:")
    
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Flatten the images and transpose
    X_train = X_train.reshape(X_train.shape[0], -1).T  # Now shape is (784, n_samples)
    
    # Convert labels to one-hot encoding and transpose
    num_classes = 10
    Y_train = np.eye(num_classes)[y_train].T  # Now shape is (10, n_samples)
    
    print("Input data shape:", X_train.shape)
    print("Labels shape:", Y_train.shape)
    
    # Create a simple neural network
    model = Sequential([
        Dense(200, activation='sigmoid', input_shape=(784,)),
        Dense(200, activation='sigmoid'),
        Dense(10, activation='softmax')
    ])
    
    # Set hyperparameters
    learning_rate = 0.1
    loss = 'categorical_crossentropy'
    batch_size = 8
    
    # Create SGD optimizer
    optimizer = SGD(learning_rate=learning_rate)
    
    # Calculate initial accuracy
    initial_acc = get_accuracy(model, X_train, Y_train)
    print(f"Initial accuracy: {initial_acc:.4f}")
    
    minimize(optimizer, model, X_train, Y_train, loss, batch_size)
    
    # Calculate final accuracy
    final_acc = get_accuracy(model, X_train, Y_train)
    print(f"\nFinal accuracy: {final_acc:.4f}")
    
    # Assert that accuracy improved
    assert final_acc > initial_acc, "Model accuracy did not improve"
    
    # Assert that final accuracy is above 60%
    assert final_acc > 0.6, f"Final accuracy {final_acc:.4f} is below 60%"
    
    print("\nMinimize function test passed!")
