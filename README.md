# MicroKeras

MicroKeras is a minimal implementation of a Keras-like deep-learning library, built from scratch using Python and NumPy. It provides a simple and intuitive API for building, training, and evaluating neural networks similar to the Keras API.

## Features

- Sequential model API
- Dense (fully connected) layers
- Various activation functions (ReLU, Sigmoid, Softmax)
- Loss functions (Mean Squared Error, Categorical Cross-Entropy)
- Optimizers (Stochastic Gradient Descent)
- Dataset loaders (MNIST, California Housing)
- Model saving and loading

## Installation

To install MicroKeras, clone this repository and install the required dependencies:

```bash
# TBD
```

## Usage Examples

### MNIST Classification

Here's an example of how to use MicroKeras for classifying MNIST digits:

```python
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import mnist

# Load and preprocess MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Create and compile the model
model = Sequential([
    Dense(200, activation='sigmoid', input_shape=(784,)),
    Dense(200, activation='sigmoid'),
    Dense(10, activation='softmax')
])
model.compile(optimizer=SGD(learning_rate=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=10)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
```

### California Housing Regression

Here's an example of using MicroKeras for regression on the California Housing dataset:

```python
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import california_housing

# Load and preprocess California Housing data
(X_train, y_train), (X_test, y_test) = california_housing.load_data()
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Create and compile the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(8,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer=SGD(learning_rate=0.01),
              loss='mean_squared_error',
              metrics=[])

# Train the model
history = model.fit(X_train, y_train, batch_size=16, epochs=100)

# Evaluate the model
test_mse = model.evaluate(X_test, y_test)
print(f"Test MSE: {test_mse:.4f}")
```

## Running Tests

To ensure the library is functioning correctly, you can run the included tests:

1. Run all tests:
   ```bash
   pytest -v -s tests
   ```

2. Run a specific test:
   ```bash
   pytest -v -s tests/test_sigmoid.py
   ```

## Project Structure

```
microkeras/
├── examples/
├── microkeras/
│   ├── activations/
│   ├── datasets/
│   ├── layers/
│   ├── losses/
│   ├── models/
│   ├── operations/
│   └── optimizers/
├── tests/
├── README.md
├── requirements.txt
└── setup.py
```
