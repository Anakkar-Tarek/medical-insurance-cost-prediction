"""
Medical Insurance Cost Prediction - Training Script

This script trains 4 models and saves the best one
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_and_prepare_data(filepath='data/insurance.csv'):
    """
    Load and prepare the dataset for training.
    
    Returns:
        X_train, X_test, y_train, y_test, preprocessors
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    
    # Separate features and target
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['sex', 'smoker', 'region']
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale numerical features (optional, but good practice)
    scaler = StandardScaler()
    numerical_cols = ['age', 'bmi', 'children']
    
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    preprocessors = {
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }
    
    return X_train, X_test, y_train, y_test, preprocessors


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance.
    
    Returns:
        Dictionary with metrics
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  R²:   {r2:.4f}")
    
    return {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse': mse
    }


def train_linear_regression(X_train, y_train, X_test, y_test):
    """Train baseline Linear Regression."""
    print("\n" + "="*60)
    print("Training Linear Regression (Baseline)")
    print("="*60)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    metrics = evaluate_model(model, X_test, y_test, "Linear Regression")
    
    return model, metrics


def train_ridge_regression(X_train, y_train, X_test, y_test):
    """Train Ridge Regression with hyperparameter tuning."""
    print("\n" + "="*60)
    print("Training Ridge Regression")
    print("="*60)
    
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    }
    
    ridge = Ridge(random_state=RANDOM_SEED)
    grid_search = GridSearchCV(
        ridge, param_grid, cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    print("Performing GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test, "Ridge Regression")
    metrics['best_params'] = grid_search.best_params_
    
    return best_model, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with hyperparameter tuning."""
    print("\n" + "="*60)
    print("Training Random Forest Regressor")
    print("="*60)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    print("Performing GridSearchCV (this may take a minute)...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test, "Random Forest")
    metrics['best_params'] = grid_search.best_params_
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return best_model, metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with hyperparameter tuning."""
    print("\n" + "="*60)
    print("Training XGBoost Regressor")
    print("="*60)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1)
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    print("Performing GridSearchCV (this may take a minute)...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test, "XGBoost")
    metrics['best_params'] = grid_search.best_params_
    
    return best_model, metrics


def save_model_and_artifacts(model, preprocessors, metrics, filepath='models/'):
    """Save the trained model and preprocessing artifacts."""
    import os
    os.makedirs(filepath, exist_ok=True)
    
    # Save model
    model_path = f"{filepath}best_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    # Save preprocessors
    prep_path = f"{filepath}preprocessors.pkl"
    with open(prep_path, 'wb') as f:
        pickle.dump(preprocessors, f)
    print(f"Preprocessors saved to: {prep_path}")
    
    # Save metrics
    metrics_path = f"{filepath}metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("Medical Insurance Cost Prediction - Model Training")
    print("="*60)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, preprocessors = load_and_prepare_data()
    
    # Train all models
    results = []
    
    # 1. Linear Regression
    lr_model, lr_metrics = train_linear_regression(X_train, y_train, X_test, y_test)
    results.append((lr_model, lr_metrics))
    
    # 2. Ridge Regression
    ridge_model, ridge_metrics = train_ridge_regression(X_train, y_train, X_test, y_test)
    results.append((ridge_model, ridge_metrics))
    
    # 3. Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    results.append((rf_model, rf_metrics))
    
    # 4. XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    results.append((xgb_model, xgb_metrics))
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_df = pd.DataFrame([m for _, m in results])
    comparison_df = comparison_df.sort_values('rmse')
    print(comparison_df[['model_name', 'rmse', 'mae', 'r2']])
    
    # Select best model (lowest RMSE)
    best_model, best_metrics = min(results, key=lambda x: x[1]['rmse'])
    
    print(f"\n🏆 Best Model: {best_metrics['model_name']}")
    print(f"   RMSE: ${best_metrics['rmse']:,.2f}")
    print(f"   R²: {best_metrics['r2']:.4f}")
    
    # Save best model
    save_model_and_artifacts(best_model, preprocessors, best_metrics)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()