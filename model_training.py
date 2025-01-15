import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import joblib

def train_model():
    """
    Train the phishing detection model
    """
    # Load preprocessed data
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    
    # Initialize the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Save the model
    print("Saving model...")
    joblib.dump(model, 'models/phishing_detector.joblib')
    
    return model

if __name__ == "__main__":
    model = train_model()
    
    # Quick evaluation on training data
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    
    print("\nTraining Set Performance:")
    y_pred = model.predict(X_train)
    print(classification_report(y_train, y_pred))
