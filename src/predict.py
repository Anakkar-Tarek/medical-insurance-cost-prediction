"""
Prediction script for medical insurance cost estimation.
Usage: python predict.py --age 30 --sex male --bmi 25 --children 1 --smoker no --region northwest
"""

import pickle
import pandas as pd
import argparse

def load_artifacts():
    """Load model and preprocessors."""
    with open('../models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('../models/preprocessors.pkl', 'rb') as f:
        preprocessors = pickle.load(f)
    return model, preprocessors

def make_prediction(model, preprocessors, input_dict):
    """Make a single prediction."""
    df = pd.DataFrame([input_dict])
    
    # Encode categoricals
    for col, encoder in preprocessors['label_encoders'].items():
        df[col] = encoder.transform(df[col])
    
    # Scale numericals
    numerical_cols = ['age', 'bmi', 'children']
    df[numerical_cols] = preprocessors['scaler'].transform(df[numerical_cols])
    
    prediction = model.predict(df)[0]
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--age', type=int, required=True)
    parser.add_argument('--sex', type=str, required=True)
    parser.add_argument('--bmi', type=float, required=True)
    parser.add_argument('--children', type=int, required=True)
    parser.add_argument('--smoker', type=str, required=True)
    parser.add_argument('--region', type=str, required=True)
    
    args = parser.parse_args()
    
    model, preprocessors = load_artifacts()
    
    input_data = {
        'age': args.age,
        'sex': args.sex,
        'bmi': args.bmi,
        'children': args.children,
        'smoker': args.smoker,
        'region': args.region
    }
    
    prediction = make_prediction(model, preprocessors, input_data)
    print(f"\n💰 Predicted Insurance Charges: ${prediction:,.2f}\n")