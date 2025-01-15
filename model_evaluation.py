import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model():
    """
    Evaluate the trained model on test data
    """
    # Load test data
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    # Load the model
    model = joblib.load('models/phishing_detector.joblib')
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('evaluation_results.png')
    plt.close()
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == "__main__":
    evaluate_model()
