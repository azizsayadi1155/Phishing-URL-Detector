import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the phishing URL dataset
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Remove unnecessary columns
    columns_to_drop = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title']
    df = df.drop(columns=columns_to_drop)
    
    # Split features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Create necessary directories
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Process the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/phishing_dataset.csv')
    
    # Save processed data
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
