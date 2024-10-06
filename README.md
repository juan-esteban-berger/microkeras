# MicroKeras

MicroKeras is a minimal implementation of the Sequential Class from the Keras deep-learning library, built from scratch using Python and NumPy. It provides a simple and intuitive API for building, training, and evaluating neural networks inspired by the Keras API.

## Documentation

For full documentation, please visit [https://microkeras.readthedocs.io/en/latest/](https://microkeras.readthedocs.io/en/latest/)

## Installation

To install MicroKeras, use pip:

```bash
pip install microkeras
```

## Features

MicroKeras offers the following features:

- Sequential model API
- Dense (fully connected) layers
- Various activation functions (ReLU, Sigmoid, Softmax, Linear)
- Loss functions (Mean Squared Error, Categorical Cross-Entropy)
- Optimizers (Stochastic Gradient Descent)
- Dataset loaders (MNIST, California Housing)
- Model saving and loading

## Quick Start: MNIST Classification Example

Let's walk through a complete example using MicroKeras to classify handwritten digits from the MNIST dataset. This example will demonstrate how to load data, create a model, train it, make predictions, and save/load the model.

### Importing Dependencies

First, let's import the necessary modules:

```python
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import mnist
```

### Loading and Preprocessing Data

Next, we'll load the MNIST dataset and preprocess it:

```python
# Load and preprocess MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize the input data
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# One-hot encode the labels
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

print("Data shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
```

### Creating the Model

Now, let's create our Sequential model:

```python
model = Sequential([
    Dense(200, activation='sigmoid', input_shape=(784,)),
    Dense(200, activation='sigmoid'),
    Dense(10, activation='softmax')
])

model.summary()
```

This model consists of two hidden layers with sigmoid activation and an output layer with softmax activation, suitable for multi-class classification.

### Compiling the Model

We'll compile the model using Stochastic Gradient Descent (SGD) as the optimizer and Categorical Cross-Entropy as the loss function:

```python
optimizer = SGD(learning_rate=0.1)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### Training the Model

Let's train the model for 10 epochs with a batch size of 32:

```python
history = model.fit(X_train,
                    y_train,
                    batch_size=32,
                    epochs=10,
                    validation_split=0.1)

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()
```

### Evaluating the Model

After training, we can evaluate the model on the test set:

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
```

### Making Predictions

Let's make predictions for the first 5 test samples:

```python
predictions = model.predict(X_test[:5])
print("Predictions for the first 5 test samples:")
print(np.argmax(predictions, axis=1))
print("Actual labels:")
print(np.argmax(y_test[:5], axis=1))

# Visualize the predictions
plt.figure(figsize=(12, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {np.argmax(predictions[i])}\nTrue: {np.argmax(y_test[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

### Saving and Loading the Model

MicroKeras allows you to save and load models:

```python
# Save the model
model.save('mnist_model.json')

# Load the model
loaded_model = Sequential.load('mnist_model.json')

# Compile the loaded model
loaded_model.compile(optimizer=optimizer,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Evaluate the loaded model
loaded_test_loss, loaded_test_accuracy = loaded_model.evaluate(X_test, y_test)
print(f"Loaded model test accuracy: {loaded_test_accuracy:.4f}")
```

### Viewing Training History

Finally, let's print out the training history:

```python
print("\nTraining History:")
print("Epoch\tAccuracy\tLoss")
for epoch, (accuracy, loss) in enumerate(zip(history['accuracy'], history['loss']), 1):
    print(f"{epoch}\t{accuracy:.4f}\t{loss:.4f}")
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
