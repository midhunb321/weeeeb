from flask import render_template, request
import numpy as np
import joblib
from app import app

# Load the trained model (ensure that scikit-learn version is compatible)
model = joblib.load('app/gb_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract feature values from the form
        age = int(request.form['age'])
        overall_rating = int(request.form['overall_rating'])
        positions_encoded = int(request.form['positions_encoded'])
        
        # Make prediction using the loaded model
        input_features = np.array([[age, overall_rating, positions_encoded]])
        predicted_wage = model.predict(input_features)[0]
        
        return render_template('result.html', predicted_wage=predicted_wage)
    else:
        return render_template('index.html')
