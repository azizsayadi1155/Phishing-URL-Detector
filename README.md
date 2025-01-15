# Phishing-URL-Detector

## Dataset Download Link:
Phishing URLs from UCI: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
## Project Structure

**Data Preprocessing (data_preprocessing.py):**

* Loads the raw dataset
* Removes unnecessary columns
* Scales the features using StandardScaler
* Splits data into training and test sets
* Saves processed data and scaler for later use

**Model Training (model_training.py):**

* Uses RandomForestClassifier for good performance and interpretability
* Trains on the preprocessed data
* Saves the trained model
* Prints initial training performance metrics

**Model Evaluation (model_evaluation.py):**

* Evaluates model on test data
* Generates confusion matrix visualization
* Creates feature importance plot
* Saves evaluation results as images

**URL Feature Extraction (url_features.py):**

* Extracts relevant features from new URLs
* Handles both static URL analysis and dynamic webpage analysis
* Transforms features to match the model's expected input format

**Flask Application (app.py):**

* Provides a web interface for URL checking
* Handles URL submissions
* Returns predictions with confidence scores

**Web Interface (templates/index.html):**

* Clean, responsive design using Bootstrap
* Real-time feedback on URL analysis
* Visual indicators for safe/dangerous URLs
* Displays confidence scores

**Data Visualization (data_visualization.py):**

* Class Distribution
* URL Length Analysis
* Feature Correlations
* TLD Analysis
* Security Features
* Special Characters Analysis
* Domain Analysis
* Interactive Scatter Plot

## How to run the project:

1. Download the dataset and save it in data/phishing_dataset.csv
2. install the requirements: pip install pandas numpy scikit_learn pickle joblib seaborn matplotlib flask plotly re urllib.parse tldextract requests bs4
3. run data_preprocessing.py
4. run data_visualization.py
5. run model_training.py
6. run model_evaluation.py
7. run url_features.py
8. run app.py
