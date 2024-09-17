import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_data():
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
