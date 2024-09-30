import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data():
    """
    Load and preprocess the California Housing dataset.

    This function loads the California Housing dataset from a parquet file,
    splits it into training and testing sets, and standardizes the feature values.

    Returns:
        tuple: A tuple containing two pairs:
            ((X_train, y_train), (X_test, y_test))
            where:
            X_train (numpy.ndarray): Training features, standardized.
            y_train (numpy.ndarray): Training target values.
            X_test (numpy.ndarray): Testing features, standardized.
            y_test (numpy.ndarray): Testing target values.

    The features are standardized to have zero mean and unit variance.
    The target variable 'MedVal' (median house value) is not standardized.

    Note:
        - The dataset is split with 80% for training and 20% for testing.
        - A random state of 42 is used for reproducibility.
    """
    file_path = os.path.join(os.path.dirname(__file__), 'california_housing.parquet')
    
    df = pd.read_parquet(file_path)
    
    # Separate features and target
    X = df.drop('MedVal', axis=1).values
    y = df['MedVal'].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_test, y_test)
