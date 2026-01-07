"""
Flask API for medical insurance cost prediction.
"""

from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model at startup (only once)
print("Loading model...")
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/preprocessors.pkl', 'rb') as f:
    preprocessors = pickle.load(f)
print("✅ Model loaded!")

@app.route('/')
def home():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Medical Insurance Cost Prediction API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Make a prediction',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/health')
def health():
    """Health check."""
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict insurance charges.
    
    Expected JSON:
    {
        "age": 30,
        "sex": "male",
        "bmi": 25.0,
        "children": 1,
        "smoker": "no",
        "region": "northwest"
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess
        for col, encoder in preprocessors['label_encoders'].items():
            df[col] = encoder.transform(df[col])
        
        numerical_cols = ['age', 'bmi', 'children']
        df[numerical_cols] = preprocessors['scaler'].transform(df[numerical_cols])
        
        # Predict
        prediction = model.predict(df)[0]
        
        # Return result
        return jsonify({
            'prediction': float(prediction),
            'formatted': f"${prediction:,.2f}",
            'input': data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)