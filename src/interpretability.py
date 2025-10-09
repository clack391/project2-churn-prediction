"""
Model Interpretability using SHAP
Provides global and local explanations for model predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os

def load_models_and_data(model_dir='models', data_dir='data/processed'):
    """Load trained models and test data"""
    logistic = joblib.load(os.path.join(model_dir, 'logistic.pkl'))
    rf = joblib.load(os.path.join(model_dir, 'rf.pkl'))
    xgb_model = joblib.load(os.path.join(model_dir, 'xgb.pkl'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    
    # Load test data
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    X_test = test[feature_names]
    y_test = test['Churn']
    
    print(f"Loaded models and test data: {len(X_test)} samples, {len(feature_names)} features")
    
    return logistic, rf, xgb_model, X_test, y_test, feature_names

def get_logistic_feature_importance(model, X_train, feature_names):
    """
    Get feature importance for Logistic Regression using coefficients
    Formula: importance = |coefficient * std_dev_of_feature|
    """
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION FEATURE IMPORTANCE")
    print("="*60)
    
    # Get coefficients
    coefficients = model.coef_[0]
    
    # Calculate feature standard deviations
    feature_std = X_train.std()
    
    # Calculate importance: |coefficient * std_dev|
    importance = np.abs(coefficients * feature_std.values)
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return feature_importance

def create_shap_explainer_tree(model, X_sample, model_name):
    """Create SHAP TreeExplainer for tree-based models"""
    print(f"\nCreating SHAP TreeExplainer for {model_name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    print(f"SHAP values computed for {len(X_sample)} samples")
    
    return explainer, shap_values

def plot_global_feature_importance(shap_values, X_sample, feature_names, model_name, output_dir='data/processed'):
    """Plot global feature importance using SHAP"""
    print(f"\nPlotting global feature importance for {model_name}...")
    
    # Get mean absolute SHAP values for ranking
    if hasattr(shap_values, 'values'):
        vals = shap_values.values
    else:
        vals = shap_values
    
    # Handle different shapes
    if len(vals.shape) == 3:
        vals = vals[:, :, -1]  # Take positive class for binary classification
    
    # Compute mean absolute values
    mean_abs_shap = np.abs(vals).mean(axis=0)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=True)  # Ascending for horizontal bar plot
    
    # Take top 15
    top_features = importance_df.tail(15)
    
    # Summary plot (bar) - manual creation for better control
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Mean |SHAP Value| (average impact on model output)', fontsize=11)
    plt.title(f'{model_name} - Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f'shap_importance_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{filename}")
    plt.close()
    
    # Summary plot (beeswarm) - shows feature values
    plt.figure(figsize=(10, 8))
    try:
        # Try to use shap.summary_plot
        if hasattr(shap_values, 'values'):
            shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
        else:
            # For older SHAP format, create explanation object
            if isinstance(shap_values, list):
                # Binary classification - take positive class
                vals_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                vals_to_plot = shap_values
            shap.summary_plot(vals_to_plot, X_sample, show=False, max_display=15, plot_type="dot")
        
        plt.title(f'{model_name} - SHAP Summary Plot', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'shap_summary_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/{filename}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create beeswarm plot: {e}")
        plt.close()

def plot_logistic_importance(feature_importance, output_dir='data/processed'):
    """Plot feature importance for Logistic Regression"""
    print("\nPlotting Logistic Regression feature importance...")
    
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    
    # Color code by positive/negative coefficient
    colors = ['green' if c > 0 else 'red' for c in top_features['coefficient']]
    
    plt.barh(range(len(top_features)), top_features['importance'], color=colors, alpha=0.7, edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance (|Coefficient × Std Dev|)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Logistic Regression - Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Positive coefficient (↑ churn)'),
        Patch(facecolor='red', alpha=0.7, label='Negative coefficient (↓ churn)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'logistic_importance.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/logistic_importance.png")
    plt.close()

def compare_feature_importance_across_models(logistic_importance, rf_shap_values, xgb_shap_values, 
                                             X_sample, feature_names, output_dir='data/processed'):
    """Create a comparison of top features across all models"""
    print("\nComparing feature importance across models...")
    
    # Get top features from each model
    logistic_top = logistic_importance.head(10)['feature'].tolist()
    
    # For SHAP, get mean absolute SHAP values
    # Handle different SHAP value formats
    if hasattr(rf_shap_values, 'values'):
        rf_vals = rf_shap_values.values
    else:
        rf_vals = rf_shap_values
    
    if hasattr(xgb_shap_values, 'values'):
        xgb_vals = xgb_shap_values.values
    else:
        xgb_vals = xgb_shap_values
    
    # Debug: Check shapes
    print(f"RF SHAP values shape: {rf_vals.shape}")
    print(f"XGB SHAP values shape: {xgb_vals.shape}")
    print(f"Feature names length: {len(feature_names)}")
    
    # For SHAP Explanation objects, extract the values correctly
    # SHAP returns (n_samples, n_features) or (n_samples, n_features, n_outputs)
    if len(rf_vals.shape) == 3:
        # Take the last dimension (for binary classification, typically [:, :, 1])
        rf_vals = rf_vals[:, :, -1]
    if len(xgb_vals.shape) == 3:
        xgb_vals = xgb_vals[:, :, -1]
    
    # Now compute mean across samples (axis=0)
    rf_mean_importance = np.abs(rf_vals).mean(axis=0)
    xgb_mean_importance = np.abs(xgb_vals).mean(axis=0)
    
    print(f"RF mean importance shape: {rf_mean_importance.shape}")
    print(f"XGB mean importance shape: {xgb_mean_importance.shape}")
    
    # Ensure 1D - flatten completely
    rf_mean_importance = np.atleast_1d(rf_mean_importance).flatten()
    xgb_mean_importance = np.atleast_1d(xgb_mean_importance).flatten()
    
    print(f"After flatten - RF: {rf_mean_importance.shape}, XGB: {xgb_mean_importance.shape}")
    
    # Final check - ensure lengths match
    if len(rf_mean_importance) != len(feature_names):
        print(f"ERROR: RF importance length {len(rf_mean_importance)} != features {len(feature_names)}")
        print(f"RF importance type: {type(rf_mean_importance)}")
        print(f"RF importance: {rf_mean_importance}")
        # Skip comparison if there's a mismatch
        return None
    
    if len(xgb_mean_importance) != len(feature_names):
        print(f"ERROR: XGB importance length {len(xgb_mean_importance)} != features {len(feature_names)}")
        # Skip comparison if there's a mismatch
        return None
    
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_mean_importance
    }).sort_values('importance', ascending=False)
    rf_top = rf_importance.head(10)['feature'].tolist()
    
    xgb_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_mean_importance
    }).sort_values('importance', ascending=False)
    xgb_top = xgb_importance.head(10)['feature'].tolist()
    
    # Find common top features
    all_top = set(logistic_top + rf_top + xgb_top)
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Feature': list(all_top),
        'Logistic (Rank)': [logistic_top.index(f)+1 if f in logistic_top else np.nan for f in all_top],
        'RF (Rank)': [rf_top.index(f)+1 if f in rf_top else np.nan for f in all_top],
        'XGBoost (Rank)': [xgb_top.index(f)+1 if f in xgb_top else np.nan for f in all_top]
    })
    
    # Calculate average rank
    comparison['Avg Rank'] = comparison[['Logistic (Rank)', 'RF (Rank)', 'XGBoost (Rank)']].mean(axis=1)
    comparison = comparison.sort_values('Avg Rank')
    
    print("\n" + "="*60)
    print("TOP FEATURES ACROSS ALL MODELS")
    print("="*60)
    print(comparison.head(15).to_string(index=False))
    
    # Save
    comparison.to_csv(os.path.join(output_dir, 'feature_importance_comparison.csv'), index=False)
    print(f"\nComparison saved to {output_dir}/feature_importance_comparison.csv")
    
    return comparison

def save_explainers(rf_explainer, xgb_explainer, logistic_importance, model_dir='models'):
    """Save SHAP explainers and importance data for app use"""
    print("\nSaving explainers for app...")
    
    # Save explainers (they're lightweight for tree models)
    joblib.dump(rf_explainer, os.path.join(model_dir, 'rf_explainer.pkl'))
    joblib.dump(xgb_explainer, os.path.join(model_dir, 'xgb_explainer.pkl'))
    
    # Save logistic importance
    joblib.dump(logistic_importance, os.path.join(model_dir, 'logistic_importance.pkl'))
    
    print(f"Explainers saved to {model_dir}/")

def demonstrate_local_explanation(model, explainer, X_sample, idx, model_name, feature_names):
    """
    Demonstrate local explanation for a single prediction
    This shows how to use SHAP for individual customer explanations
    """
    print(f"\n" + "="*60)
    print(f"LOCAL EXPLANATION EXAMPLE - {model_name}")
    print("="*60)
    
    # Get single sample
    sample = X_sample.iloc[[idx]]
    
    # Get prediction
    pred_proba = model.predict_proba(sample)[0, 1]
    pred_class = model.predict(sample)[0]
    
    print(f"\nSample index: {idx}")
    print(f"Predicted churn probability: {pred_proba*100:.1f}%")
    print(f"Predicted class: {'Churn' if pred_class == 1 else 'No Churn'}")
    
    # Get SHAP values for this sample
    if hasattr(explainer, 'shap_values'):
        # For tree models
        shap_values_sample = explainer.shap_values(sample)
        if isinstance(shap_values_sample, list):
            shap_values_sample = shap_values_sample[1]  # Get positive class
    else:
        # For newer SHAP version
        shap_values_sample = explainer(sample)
        if hasattr(shap_values_sample, 'values'):
            shap_values_sample = shap_values_sample.values[0]
    
    # Show top contributing features
    if len(shap_values_sample.shape) > 1:
        shap_values_sample = shap_values_sample[0]
    
    feature_contributions = pd.DataFrame({
        'feature': feature_names,
        'value': sample.values[0],
        'shap_value': shap_values_sample
    }).sort_values('shap_value', key=abs, ascending=False)
    
    print("\nTop 5 features pushing prediction:")
    print(feature_contributions.head(5).to_string(index=False))
    
    print("="*60)

def run_interpretability_pipeline():
    """Main interpretability pipeline"""
    print("="*60)
    print("MODEL INTERPRETABILITY PIPELINE")
    print("="*60)
    
    # 1. Load models and data
    logistic, rf, xgb_model, X_test, y_test, feature_names = load_models_and_data()
    
    # 2. Sample data for SHAP (use subset to speed up)
    # SHAP can be slow on large datasets, so we sample
    sample_size = min(200, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    print(f"\nUsing {sample_size} samples for SHAP analysis")
    
    # Also load train data for logistic regression std calculation
    train = pd.read_csv('data/processed/train.csv')
    X_train = train[feature_names]
    
    # 3. Logistic Regression - Coefficient Analysis
    logistic_importance = get_logistic_feature_importance(logistic, X_train, feature_names)
    plot_logistic_importance(logistic_importance)
    
    # 4. Random Forest - SHAP Analysis
    print("\n" + "="*60)
    print("RANDOM FOREST SHAP ANALYSIS")
    print("="*60)
    rf_explainer, rf_shap_values = create_shap_explainer_tree(rf, X_sample, "Random Forest")
    plot_global_feature_importance(rf_shap_values, X_sample, feature_names, "Random Forest")
    
    # 5. XGBoost - SHAP Analysis
    print("\n" + "="*60)
    print("XGBOOST SHAP ANALYSIS")
    print("="*60)
    xgb_explainer, xgb_shap_values = create_shap_explainer_tree(xgb_model, X_sample, "XGBoost")
    plot_global_feature_importance(xgb_shap_values, X_sample, feature_names, "XGBoost")
    
    # 6. Compare feature importance across models
    comparison = compare_feature_importance_across_models(
        logistic_importance, rf_shap_values, xgb_shap_values, X_sample, feature_names
    )
    
    # Only proceed if comparison was successful
    if comparison is not None:
        demonstrate_local_explanation(xgb_model, xgb_explainer, X_sample, 0, "XGBoost", feature_names)
    
    # 8. Save explainers for app
    save_explainers(rf_explainer, xgb_explainer, logistic_importance)
    
    print("\n" + "="*60)
    print("INTERPRETABILITY ANALYSIS COMPLETE!")
    print("="*60)
    
    return rf_explainer, xgb_explainer, logistic_importance, comparison

if __name__ == "__main__":
    rf_explainer, xgb_explainer, logistic_importance, comparison = run_interpretability_pipeline()