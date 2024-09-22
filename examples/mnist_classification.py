import numpy as np
from microkeras.models import Sequential
from microkeras.layers import Dense
from microkeras.optimizers import SGD
from microkeras.datasets import mnist

# Load and preprocess MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize the input data
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# One-hot encode the labels
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Create the model
model = Sequential([
    Dense(200, activation='sigmoid', input_shape=(784,)),
    Dense(200, activation='sigmoid'),
    Dense(10, activation='softmax')
])

# Compile the model
optimizer = SGD(learning_rate=0.1)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train,
                    y_train,
                    batch_size=32,
                    epochs=10)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Make predictions
predictions = model.predict(X_test[:5])
print("Predictions for the first 5 test samples:")
print(np.argmax(predictions, axis=1))
print("Actual labels:")
print(np.argmax(y_test[:5], axis=1))

# Save the model
model.save('example_models/mnist_model.json')

# Load the model
loaded_model = Sequential.load('example_models/mnist_model.json')

# Compile the loaded model
loaded_model.compile(optimizer=optimizer,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Evaluate the loaded model
loaded_test_loss, loaded_test_accuracy = loaded_model.evaluate(X_test, y_test)
print(f"Loaded model test accuracy: {loaded_test_accuracy:.4f}")

# Print training history
print("\nTraining History:")
print("Epoch\tAccuracy\tLoss")
for epoch, (accuracy, loss) in enumerate(zip(history['accuracy'], history['loss']), 1):
    print(f"{epoch}\t{accuracy:.4f}\t{loss:.4f}")
