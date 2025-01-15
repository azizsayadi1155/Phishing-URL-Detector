from flask import Flask, request, render_template, jsonify
import joblib
from url_features import URLFeatureExtractor

app = Flask(__name__)

# Load the model
model = joblib.load('models/phishing_detector.joblib')
feature_extractor = URLFeatureExtractor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        url = request.form['url']
        
        # Extract features
        features = feature_extractor.extract_features(url)
        
        # Transform features
        X = feature_extractor.transform_features(features)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        result = {
            'url': url,
            'is_phishing': bool(prediction),
            'confidence': float(max(probability)),
            'status': 'success'
        }
        
    except Exception as e:
        result = {
            'url': url,
            'error': str(e),
            'status': 'error'
        }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
