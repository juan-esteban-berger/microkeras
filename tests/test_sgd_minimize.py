import pytest
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.optimizers.minimize import minimize
from microkeras.optimizers.select_batch import select_batches
from microkeras.optimizers.gradient_descent import gradient_descent

def test_sgd_minimize():
    print()
    print("Minimize function test:")

    # Create a simple neural network
    model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(5,)),
        Dense(2, activation='softmax')
    ])

    # Create sample input data and labels
    X_train = np.array([[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9],
                        [0.2, 0.3, 0.4],
                        [0.5, 0.6, 0.7]])  # (5, 3) shape
    Y_train = np.array([[1, 0, 1],
                        [0, 1, 0]])  # (2, 3) shape

    print("Input data shape:", X_train.shape)
    print("Labels shape:", Y_train.shape)

    # Set hyperparameters
    learning_rate = 0.1
    loss = 'categorical_crossentropy'
    num_iterations = 5
    batch_size = 2

    # Create SGD optimizer
    optimizer = SGD(learning_rate=learning_rate)

    # Create two copies of the model
    model1 = model.copy()
    model2 = model.copy()

    # Perform minimize on model1
    minimize(optimizer, model1, X_train, Y_train, loss, num_iterations, batch_size)

    # Perform manual iterations on model2
    for _ in range(num_iterations):
        X_batch, Y_batch = select_batches(X_train, Y_train, batch_size)
        gradient_descent(model2, X_batch, Y_batch, loss, learning_rate)

    # Compare gradients
    print("\nComparing gradients:")
    for i, (layer1, layer2) in enumerate(zip(model1.layers, model2.layers)):
        print(f"\nLayer {i}:")
        np.testing.assert_allclose(layer1.W, layer2.W, rtol=1e-5, atol=1e-5, 
                                   err_msg=f"Weights in layer {i} do not match")
        np.testing.assert_allclose(layer1.b, layer2.b, rtol=1e-5, atol=1e-5, 
                                   err_msg=f"Biases in layer {i} do not match")
        print(f"  Weights match: {np.allclose(layer1.W, layer2.W, rtol=1e-5, atol=1e-5)}")
        print(f"  Biases match: {np.allclose(layer1.b, layer2.b, rtol=1e-5, atol=1e-5)}")

    print("\nMinimize function test passed!")
