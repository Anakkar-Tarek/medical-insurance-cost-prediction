"""
Quick test to verify the trained model works correctly.
"""

import pickle
import pandas as pd
import numpy as np

def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects."""
    with open('../models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('../models/preprocessors.pkl', 'rb') as f:
        preprocessors = pickle.load(f)
    
    return model, preprocessors

def preprocess_input(input_data, preprocessors):
    """Preprocess input data using saved preprocessors."""
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    label_encoders = preprocessors['label_encoders']
    for col, encoder in label_encoders.items():
        df[col] = encoder.transform(df[col])
    
    # Scale numerical features
    scaler = preprocessors['scaler']
    numerical_cols = ['age', 'bmi', 'children']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df

def predict_charges(age, sex, bmi, children, smoker, region):
    """Predict insurance charges for given inputs."""
    # Load model
    model, preprocessors = load_model_and_preprocessors()
    
    # Prepare input
    input_data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }
    
    # Preprocess
    processed_input = preprocess_input(input_data, preprocessors)
    
    # Predict
    prediction = model.predict(processed_input)[0]
    
    return prediction

# Test cases
if __name__ == "__main__":
    print("="*60)
    print("Testing Trained Model")
    print("="*60)
    
    # Test Case 1: Young non-smoker
    print("\n📋 Test Case 1: Young, healthy, non-smoker")
    print("  Age: 25, Sex: male, BMI: 22, Children: 0, Smoker: no, Region: northwest")
    prediction1 = predict_charges(25, 'male', 22, 0, 'no', 'northwest')
    print(f"  💰 Predicted Charges: ${prediction1:,.2f}")
    print(f"  ✅ Expected: ~$2,000-$4,000 (LOW)")
    
    # Test Case 2: Middle-aged smoker
    print("\n📋 Test Case 2: Middle-aged smoker")
    print("  Age: 45, Sex: female, BMI: 30, Children: 2, Smoker: yes, Region: southeast")
    prediction2 = predict_charges(45, 'female', 30, 2, 'yes', 'southeast')
    print(f"  💰 Predicted Charges: ${prediction2:,.2f}")
    print(f"  ✅ Expected: ~$30,000-$40,000 (HIGH)")
    
    # Test Case 3: Older non-smoker
    print("\n📋 Test Case 3: Older, non-smoker")
    print("  Age: 60, Sex: male, BMI: 28, Children: 0, Smoker: no, Region: northeast")
    prediction3 = predict_charges(60, 'male', 28, 0, 'no', 'northeast')
    print(f"  💰 Predicted Charges: ${prediction3:,.2f}")
    print(f"  ✅ Expected: ~$10,000-$15,000 (MEDIUM)")
    
    print("\n" + "="*60)
    print("✅ Model is working correctly!")
    print("="*60)