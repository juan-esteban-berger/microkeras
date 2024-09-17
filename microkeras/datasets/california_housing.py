import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data():
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
