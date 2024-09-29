import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import california_housing

# Load and preprocess California Housing data
(X_train, y_train), (X_test, y_test) = california_housing.load_data()

# Reshape y to be 2D
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Create the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(8,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
optimizer = SGD(learning_rate=0.01)
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=[])  # Empty list for metrics as it's a regression problem

# Train the model
history = model.fit(X_train,
                    y_train,
                    batch_size=16,
                    epochs=100)

# Evaluate the model
test_mse = model.evaluate(X_test, y_test)
print(f"Test MSE: {test_mse:.4f}")

# Make predictions
predictions = model.predict(X_test[:5])
print("Predictions for the first 5 test samples:")
print(predictions.flatten())
print("Actual values:")
print(y_test[:5].flatten())

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

# Print training history
print("\nTraining History:")
print("Epoch\tLoss")
for epoch, loss in enumerate(history['loss'], 1):
    print(f"{epoch}\t{loss:.4f}")
