# California Housing Regression Example

<a target="_blank" href="https://colab.research.google.com/github/juan-esteban-berger/microkeras/blob/main/examples/california_housing_regression.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

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
              metrics=[])
```

## Training the Model

Let's train the model for 100 epochs with a batch size of 16:

```python
history = model.fit(X_train,
                    y_train,
                    batch_size=16,
                    epochs=100)
```

Output:
    
```
Epoch 1/100
Batch 1032/1032 - Loss: 1.1183: : 1088it [00:04, 220.54it/s]
Epoch 2/100
Batch 1032/1032 - Loss: 0.6722: : 1088it [00:05, 183.74it/s]
Epoch 3/100
Batch 1032/1032 - Loss: 1.0607: : 1088it [00:02, 417.53it/s]
Epoch 4/100
Batch 1032/1032 - Loss: 0.6741: : 1088it [00:02, 402.56it/s]
Epoch 5/100
Batch 1032/1032 - Loss: 0.5054: : 1088it [00:02, 414.89it/s]
...
Epoch 100/100
Batch 1032/1032 - Loss: 0.2898: : 1088it [00:01, 903.47it/s]
```

## Evaluating the Model

After training, we can evaluate the model on the test set:

```python
test_mse = model.evaluate(X_test, y_test)
print(f"Test MSE: {test_mse:.4f}")
```

Output:

```
Test MSE: 0.3325
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

Output:

```
Predictions for the first 5 test samples:
[0.50417322 1.0510164  4.579818   2.80260055 3.1033866 ]
Actual values:
[0.477   0.458   5.00001 2.186   2.78   ]
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

Output:

```
Loaded model test MSE: 0.3325
```

## Viewing Training History

Finally, let's print out the training history:

```python
print("\nTraining History:")
print("Epoch\tLoss")
for epoch, loss in enumerate(history['loss'], 1):
    print(f"{epoch}\t{loss:.4f}")
```

Output:

```
Training History:
Epoch	Loss
1	2.5730
2	0.6453
3	1.6912
4	0.5023
5	0.4681
...
100	0.2876
```

This example demonstrates how to use MicroKeras for a regression task, including model creation, training, evaluation, prediction, and model saving/loading.
