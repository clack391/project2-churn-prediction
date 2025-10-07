"""
Prediction utilities for the Streamlit app
Handles model loading, preprocessing, and predictions
"""

import pandas as pd
import numpy as np
import joblib
import os

def load_models(model_dir='models'):
    """Load all trained models and utilities"""
    models = {
        'logistic': joblib.load(os.path.join(model_dir, 'logistic.pkl')),
        'rf': joblib.load(os.path.join(model_dir, 'rf.pkl')),
        'xgb': joblib.load(os.path.join(model_dir, 'xgb.pkl'))
    }
    
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    
    # Load explainers
    explainers = {
        'rf': joblib.load(os.path.join(model_dir, 'rf_explainer.pkl')),
        'xgb': joblib.load(os.path.join(model_dir, 'xgb_explainer.pkl')),
        'logistic_importance': joblib.load(os.path.join(model_dir, 'logistic_importance.pkl'))
    }
    
    return models, preprocessor, feature_names, explainers

def preprocess_input(input_data, feature_names):
    """
    Preprocess user input to match training data format
    
    Args:
        input_data: dict with user inputs
        feature_names: list of expected features
    
    Returns:
        DataFrame ready for prediction
    """
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Engineer features
    df = engineer_features_for_input(df)
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select and order features
    df = df[feature_names]
    
    return df

def engineer_features_for_input(df):
    """
    Engineer features for a single input
    Must match the feature engineering in data_prep.py
    """
    df = df.copy()
    
    # tenure_bucket
    def bucket_tenure(tenure):
        if tenure <= 6:
            return 0  # '0-6m'
        elif tenure <= 12:
            return 1  # '6-12m'
        elif tenure <= 24:
            return 2  # '12-24m'
        else:
            return 3  # '24m+'
    
    if 'tenure' in df.columns:
        df['tenure_bucket'] = df['tenure'].apply(bucket_tenure)
    
    # services_count (simplified - count all services marked as 1)
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    df['services_count'] = 0
    for col in service_cols:
        if col in df.columns:
            df['services_count'] += df[col]
    
    # monthly_to_total_ratio
    if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns and 'tenure' in df.columns:
        expected_total = df['tenure'] * df['MonthlyCharges']
        df['monthly_to_total_ratio'] = df['TotalCharges'] / np.maximum(1, expected_total)
    else:
        df['monthly_to_total_ratio'] = 1.0
    
    # Flags
    if 'InternetService' in df.columns and 'TechSupport' in df.columns:
        df['internet_no_tech_support'] = ((df['InternetService'] > 0) & (df['TechSupport'] == 0)).astype(int)
    else:
        df['internet_no_tech_support'] = 0
    
    if 'InternetService' in df.columns:
        df['fiber_optic'] = (df['InternetService'] == 1).astype(int)  # Assuming fiber optic is encoded as 1
    else:
        df['fiber_optic'] = 0
    
    if 'Contract' in df.columns:
        df['month_to_month'] = (df['Contract'] == 0).astype(int)  # Month-to-month is first alphabetically
    else:
        df['month_to_month'] = 0
    
    if 'InternetService' in df.columns and 'OnlineSecurity' in df.columns:
        df['no_online_security'] = ((df['InternetService'] > 0) & (df['OnlineSecurity'] == 0)).astype(int)
    else:
        df['no_online_security'] = 0
    
    return df

def calculate_clv_for_input(input_data):
    """
    Calculate CLV for input customer
    
    Args:
        input_data: dict with MonthlyCharges, Contract, and tenure
    
    Returns:
        CLV value
    """
    monthly_charges = input_data.get('MonthlyCharges', 0)
    contract = input_data.get('Contract', 0)
    tenure = input_data.get('tenure', 0)
    
    # Calculate expected tenure based on contract type
    if contract == 0:  # Month-to-month
        expected_tenure = tenure + 6
    elif contract == 1:  # One year
        months_in_current = tenure % 12
        months_to_complete = 12 - months_in_current if months_in_current > 0 else 0
        expected_tenure = tenure + months_to_complete + 6
    else:  # Two year
        months_in_current = tenure % 24
        months_to_complete = 24 - months_in_current if months_in_current > 0 else 0
        expected_tenure = tenure + months_to_complete + 12
    
    clv = monthly_charges * expected_tenure
    
    return clv, expected_tenure

def predict_churn(input_data, model, feature_names):
    """
    Predict churn probability for a single customer
    
    Args:
        input_data: dict with customer features
        model: trained model
        feature_names: list of feature names
    
    Returns:
        churn_probability: float between 0 and 1
        prediction: 0 or 1
    """
    # Preprocess
    X = preprocess_input(input_data, feature_names)
    
    # Predict
    churn_probability = model.predict_proba(X)[0, 1]
    prediction = model.predict(X)[0]
    
    return churn_probability, prediction

def get_risk_label(probability):
    """Convert probability to risk label"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

def explain_prediction_shap(model, explainer, input_data, feature_names):
    """
    Get SHAP explanation for a prediction
    
    Args:
        model: trained model
        explainer: SHAP explainer
        input_data: dict with customer features
        feature_names: list of feature names
    
    Returns:
        DataFrame with feature contributions
    """
    # Preprocess
    X = preprocess_input(input_data, feature_names)
    
    # Get SHAP values
    shap_values = explainer(X)
    
    # Extract values
    if hasattr(shap_values, 'values'):
        shap_vals = shap_values.values[0]
    else:
        shap_vals = shap_values[0]
    
    # Create explanation dataframe
    explanation = pd.DataFrame({
        'Feature': feature_names,
        'Value': X.values[0],
        'SHAP Value': shap_vals
    })
    
    # Sort by absolute SHAP value
    explanation['Abs SHAP'] = explanation['SHAP Value'].abs()
    explanation = explanation.sort_values('Abs SHAP', ascending=False)
    
    return explanation

def explain_prediction_logistic(input_data, logistic_importance, feature_names):
    """
    Get explanation for logistic regression using coefficients
    
    Args:
        input_data: dict with customer features
        logistic_importance: DataFrame with feature importance from coefficients
        feature_names: list of feature names
    
    Returns:
        DataFrame with feature contributions
    """
    # Preprocess
    X = preprocess_input(input_data, feature_names)
    
    # Get feature values
    feature_values = X.values[0]
    
    # Calculate contribution for each feature
    contributions = []
    for feature, value in zip(feature_names, feature_values):
        if feature in logistic_importance['feature'].values:
            coef = logistic_importance[logistic_importance['feature'] == feature]['coefficient'].values[0]
            contribution = coef * value
            contributions.append({
                'Feature': feature,
                'Value': value,
                'Contribution': contribution
            })
    
    explanation = pd.DataFrame(contributions)
    explanation['Abs Contribution'] = explanation['Contribution'].abs()
    explanation = explanation.sort_values('Abs Contribution', ascending=False)
    
    return explanation

def get_ensemble_prediction(input_data, models, feature_names):
    """
    Get ensemble prediction by averaging all three models
    
    Args:
        input_data: dict with customer features
        models: dict of trained models
        feature_names: list of feature names
    
    Returns:
        average_probability: float
        individual_predictions: dict
    """
    individual_predictions = {}
    
    for model_name, model in models.items():
        prob, pred = predict_churn(input_data, model, feature_names)
        individual_predictions[model_name] = {
            'probability': prob,
            'prediction': pred
        }
    
    # Calculate average
    avg_prob = np.mean([p['probability'] for p in individual_predictions.values()])
    
    return avg_prob, individual_predictions

# Encoding mappings for user-friendly labels
# These match the LabelEncoder alphabetical sorting
ENCODING_MAPS = {
    'gender': {'Female': 0, 'Male': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'MultipleLines': {'No': 0, 'No phone service': 1, 'Yes': 2},
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingTV': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingMovies': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'PaymentMethod': {
        'Bank transfer (automatic)': 0,
        'Credit card (automatic)': 1,
        'Electronic check': 2,
        'Mailed check': 3
    }
}

def encode_user_inputs(user_inputs):
    """
    Encode user-friendly inputs to numeric values
    
    Args:
        user_inputs: dict with user-friendly values
    
    Returns:
        dict with encoded values
    """
    encoded = user_inputs.copy()
    
    for feature, mapping in ENCODING_MAPS.items():
        if feature in encoded:
            encoded[feature] = mapping.get(encoded[feature], 0)
    
    return encoded