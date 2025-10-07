"""
Model Training Pipeline
Trains Logistic Regression, Random Forest, and XGBoost with class imbalance handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import xgboost as xgb
import joblib
import os

def load_splits(data_dir='data/processed'):
    """Load train/val/test splits"""
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    print(f"Loaded data:")
    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples")
    
    return train, val, test

def prepare_features(train, val, test):
    """Prepare X and y for modeling"""
    # Features to exclude from modeling
    exclude_cols = ['customerID', 'Churn', 'CLV', 'expected_tenure']
    
    # Get feature columns
    feature_cols = [col for col in train.columns if col not in exclude_cols]
    
    # Split into X and y
    X_train = train[feature_cols]
    y_train = train['Churn']
    
    X_val = val[feature_cols]
    y_val = val['Churn']
    
    X_test = test[feature_cols]
    y_test = test['Churn']
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Target distribution (train): {y_train.value_counts(normalize=True).to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

def create_preprocessor(X_train):
    """Create and fit StandardScaler for numeric features"""
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    print("\nPreprocessor (StandardScaler) fitted")
    return scaler

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train Logistic Regression with class imbalance handling"""
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)
    
    # Use class_weight='balanced' to handle imbalance
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',  # Key for handling imbalance
        C=0.1,  # Regularization strength (tuned)
        solver='lbfgs'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    
    return model, train_metrics, val_metrics

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest with class imbalance handling"""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    # Use class_weight='balanced' to handle imbalance
    model = RandomForestClassifier(
        n_estimators=200,  # More trees for better performance
        max_depth=10,  # Limit depth to prevent overfitting
        min_samples_leaf=10,  # Larger leaves for generalization
        min_samples_split=20,
        class_weight='balanced',  # Key for handling imbalance
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    
    return model, train_metrics, val_metrics

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with class imbalance handling"""
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    
    # Calculate scale_pos_weight for imbalance handling
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,  # Balanced depth
        learning_rate=0.05,  # Lower learning rate for better convergence
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Key for handling imbalance
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    
    return model, train_metrics, val_metrics

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y, y_pred_proba)
    }
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    return metrics

def retrain_on_train_plus_val(model_class, model_params, X_train, y_train, X_val, y_val):
    """Retrain model on train+val for final model"""
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)
    
    print(f"\nRetraining on train+val: {len(X_combined)} samples")
    
    if isinstance(model_class, type):
        model = model_class(**model_params)
    else:
        model = model_class
    
    model.fit(X_combined, y_combined)
    
    return model

def save_models(logistic, rf, xgb_model, preprocessor, model_dir='models'):
    """Save trained models and preprocessor"""
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(logistic, os.path.join(model_dir, 'logistic.pkl'))
    joblib.dump(rf, os.path.join(model_dir, 'rf.pkl'))
    joblib.dump(xgb_model, os.path.join(model_dir, 'xgb.pkl'))
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))
    
    print(f"\nModels saved to {model_dir}/")

def create_comparison_table(logistic_metrics, rf_metrics, xgb_metrics):
    """Create model comparison table"""
    comparison = pd.DataFrame({
        'Logistic Regression': logistic_metrics,
        'Random Forest': rf_metrics,
        'XGBoost': xgb_metrics
    }).T
    
    # Round for display
    comparison = comparison.round(4)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON (Test Set)")
    print("="*60)
    print(comparison.to_string())
    print("="*60)
    
    return comparison

def plot_roc_curves(models, X_test, y_test, output_dir='data/processed'):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red']
    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
    
    for model, name, color in zip(models, model_names, colors):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', 
                linewidth=2, color=color)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    print(f"\nROC curves saved to {output_dir}/roc_curves.png")
    plt.close()

def plot_confusion_matrices(models, X_test, y_test, output_dir='data/processed'):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
    
    for ax, model, name in zip(axes, models, model_names):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        ax.set_title(name, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to {output_dir}/confusion_matrices.png")
    plt.close()

def test_high_risk_profile(models, feature_cols, preprocessor):
    """
    Test the high-risk profile mentioned in requirements:
    - Senior citizen with month-to-month contract
    - Fiber optic internet
    - No security/backup/tech support
    - Electronic check payment
    - Monthly charges >= $100
    Should predict >60% churn probability
    """
    print("\n" + "="*60)
    print("TESTING HIGH-RISK PROFILE")
    print("="*60)
    
    # Create a sample high-risk customer
    # This is a simplified example - you'd need to match exact encoding from training
    high_risk = pd.DataFrame({
        'gender': [1],  # Male
        'SeniorCitizen': [1],  # Yes
        'Partner': [0],  # No
        'Dependents': [0],  # No
        'tenure': [1],  # Short tenure
        'PhoneService': [1],  # Yes
        'MultipleLines': [0],  # No
        'InternetService': [1],  # Fiber optic (assuming encoding)
        'OnlineSecurity': [0],  # No
        'OnlineBackup': [0],  # No
        'DeviceProtection': [0],  # No
        'TechSupport': [0],  # No
        'StreamingTV': [0],  # No
        'StreamingMovies': [0],  # No
        'Contract': [0],  # Month-to-month (alphabetically first)
        'PaperlessBilling': [1],  # Yes
        'PaymentMethod': [1],  # Electronic check (assuming encoding)
        'MonthlyCharges': [100.0],
        'TotalCharges': [100.0],
        'tenure_bucket': [0],  # 0-6m
        'services_count': [2],
        'monthly_to_total_ratio': [1.0],
        'internet_no_tech_support': [1],
        'fiber_optic': [1],
        'month_to_month': [1],
        'no_online_security': [1]
    })
    
    # Ensure all features are present
    for col in feature_cols:
        if col not in high_risk.columns:
            high_risk[col] = 0
    
    high_risk = high_risk[feature_cols]
    
    print("\nHigh-Risk Profile Predictions:")
    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
    
    for model, name in zip(models, model_names):
        proba = model.predict_proba(high_risk)[0, 1]
        print(f"  {name}: {proba*100:.1f}% churn probability")
        
        if proba < 0.6:
            print(f"    ⚠️  Warning: Expected >60%, got {proba*100:.1f}%")
    
    print("="*60)

def run_training_pipeline():
    """Main training pipeline"""
    print("="*60)
    print("MODEL TRAINING PIPELINE")
    print("="*60)
    
    # 1. Load data
    train, val, test = load_splits()
    
    # 2. Prepare features
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_features(train, val, test)
    
    # 3. Create preprocessor
    preprocessor = create_preprocessor(X_train)
    
    # Note: For tree-based models, we don't need to scale, but we'll keep for consistency
    # and future models that might need it
    
    # 4. Train models
    logistic, logistic_train_metrics, logistic_val_metrics = train_logistic_regression(
        X_train, y_train, X_val, y_val
    )
    
    rf, rf_train_metrics, rf_val_metrics = train_random_forest(
        X_train, y_train, X_val, y_val
    )
    
    xgb_model, xgb_train_metrics, xgb_val_metrics = train_xgboost(
        X_train, y_train, X_val, y_val
    )
    
    # 5. Retrain on train+val for final models (optional but recommended)
    print("\n" + "="*60)
    print("RETRAINING ON TRAIN+VAL FOR FINAL MODELS")
    print("="*60)
    
    logistic_final = retrain_on_train_plus_val(
        LogisticRegression,
        {'max_iter': 1000, 'random_state': 42, 'class_weight': 'balanced', 'C': 0.1, 'solver': 'lbfgs'},
        X_train, y_train, X_val, y_val
    )
    
    rf_final = retrain_on_train_plus_val(
        RandomForestClassifier,
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 10, 
         'min_samples_split': 20, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
        X_train, y_train, X_val, y_val
    )
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_final = retrain_on_train_plus_val(
        xgb.XGBClassifier,
        {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': scale_pos_weight,
         'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False},
        X_train, y_train, X_val, y_val
    )
    
    # 6. Evaluate on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    logistic_test_metrics = evaluate_model(logistic_final, X_test, y_test, "Logistic Regression")
    rf_test_metrics = evaluate_model(rf_final, X_test, y_test, "Random Forest")
    xgb_test_metrics = evaluate_model(xgb_final, X_test, y_test, "XGBoost")
    
    # 7. Create comparison table
    comparison = create_comparison_table(logistic_test_metrics, rf_test_metrics, xgb_test_metrics)
    comparison.to_csv('data/processed/model_comparison.csv')
    print("\nComparison table saved to data/processed/model_comparison.csv")
    
    # 8. Plot visualizations
    models = [logistic_final, rf_final, xgb_final]
    plot_roc_curves(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    
    # 9. Test high-risk profile
    test_high_risk_profile(models, feature_cols, preprocessor)
    
    # 10. Save models
    save_models(logistic_final, rf_final, xgb_final, preprocessor)
    
    # Save feature names for later use
    joblib.dump(feature_cols, 'models/feature_names.pkl')
    print("Feature names saved to models/feature_names.pkl")
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)
    
    return models, comparison, feature_cols

if __name__ == "__main__":
    models, comparison, feature_cols = run_training_pipeline()