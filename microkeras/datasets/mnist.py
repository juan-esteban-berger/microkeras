import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_data():
    """
    Load and preprocess the MNIST dataset.

    This function loads the MNIST dataset from a parquet file,
    splits it into training and testing sets, and preprocesses the images.

    Returns:
    tuple: A tuple containing two pairs:
        - ((X_train, y_train), (X_test, y_test))
        where:
        X_train (numpy.ndarray): Training images, shape (n_samples, 28, 28), normalized.
        y_train (numpy.ndarray): Training labels, uint8 type.
        X_test (numpy.ndarray): Testing images, shape (n_samples, 28, 28), normalized.
        y_test (numpy.ndarray): Testing labels, uint8 type.

    The image data is reshaped to (28, 28) and normalized to [0, 1] range.
    Labels are converted to uint8 type.

    Note:
    - The dataset is split with 80% for training and 20% for testing.
    - A random state of 42 is used for reproducibility.
    """
    file_path = os.path.join(os.path.dirname(__file__), 'mnist.parquet')
    
    df = pd.read_parquet(file_path)
    
    # Separate features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].values.astype('uint8')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape the features to (samples, 28, 28)
    X_train = X_train.reshape(-1, 28, 28).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28).astype('float32') / 255.0

    return (X_train, y_train), (X_test, y_test)
