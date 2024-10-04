# California Housing Regression Example

This example demonstrates how to use MicroKeras for a regression task using the California Housing dataset. We'll build a model to predict housing prices based on various features.

## Importing Dependencies

First, let's import the necessary modules:

```python
import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import california_housing
```

## Loading and Preprocessing Data

Next, we'll load the California Housing dataset and preprocess it:

```python
# Load and preprocess California Housing data
(X_train, y_train), (X_test, y_test) = california_housing.load_data()

# Reshape y to be 2D
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
```

## Creating the Model

Now, let's create our Sequential model:

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(8,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
```

This model consists of three hidden layers with ReLU activation and an output layer with linear activation, suitable for regression tasks.

## Compiling the Model

We'll compile the model using Stochastic Gradient Descent (SGD) as the optimizer and Mean Squared Error (MSE) as the loss function:

```python
optimizer = SGD(learning_rate=0.01)
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=[])  # Empty list for metrics as it's a regression problem
```

## Training the Model

Let's train the model for 100 epochs with a batch size of 16:

```python
history = model.fit(X_train,
                    y_train,
                    batch_size=16,
                    epochs=100)
```

## Evaluating the Model

After training, we can evaluate the model on the test set:

```python
test_mse = model.evaluate(X_test, y_test)
print(f"Test MSE: {test_mse:.4f}")
```

## Making Predictions

Let's make predictions for the first 5 test samples:

```python
predictions = model.predict(X_test[:5])
print("Predictions for the first 5 test samples:")
print(predictions.flatten())
print("Actual values:")
print(y_test[:5].flatten())
```

## Saving and Loading the Model

MicroKeras allows you to save and load models:

```python
# Save the model
model.save('example_models/california_housing_model.json')

# Load the model
loaded_model = Sequential.load('example_models/california_housing_model.json')

# Compile the loaded model
loaded_model.compile(optimizer=optimizer,
                     loss='mean_squared_error',
                     metrics=[])

# Evaluate the loaded model
loaded_test_mse = loaded_model.evaluate(X_test, y_test)
print(f"Loaded model test MSE: {loaded_test_mse:.4f}")
```

## Viewing Training History

Finally, let's print out the training history:

```python
print("\nTraining History:")
print("Epoch\tLoss")
for epoch, loss in enumerate(history['loss'], 1):
    print(f"{epoch}\t{loss:.4f}")
```

This example demonstrates how to use MicroKeras for a regression task, including model creation, training, evaluation, prediction, and model saving/loading.
