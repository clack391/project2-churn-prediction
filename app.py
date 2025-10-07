"""
Customer Churn Prediction & CLV Analysis
Streamlit App with 3 tabs: Predict, Model Performance, CLV Overview
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Churn Prediction & CLV",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .high-risk {
        color: #d62728;
        font-weight: bold;
    }
    .medium-risk {
        color: #ff7f0e;
        font-weight: bold;
    }
    .low-risk {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CACHING FUNCTIONS
# ============================================================

@st.cache_data
def load_processed_data():
    """Load processed train/val/test data"""
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')
    return train, val, test

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {
        'Logistic Regression': joblib.load('models/logistic.pkl'),
        'Random Forest': joblib.load('models/rf.pkl'),
        'XGBoost': joblib.load('models/xgb.pkl')
    }
    feature_names = joblib.load('models/feature_names.pkl')
    return models, feature_names

@st.cache_resource
def load_explainers():
    """Load SHAP explainers"""
    rf_explainer = joblib.load('models/rf_explainer.pkl')
    xgb_explainer = joblib.load('models/xgb_explainer.pkl')
    logistic_importance = joblib.load('models/logistic_importance.pkl')
    return rf_explainer, xgb_explainer, logistic_importance

@st.cache_data
def load_model_comparison():
    """Load model comparison metrics"""
    return pd.read_csv('data/processed/model_comparison.csv', index_col=0)

@st.cache_data
def load_clv_insights():
    """Load CLV insights"""
    with open('data/processed/clv_insights.txt', 'r') as f:
        return f.read()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def engineer_features_for_input(df):
    """Engineer features for input data"""
    df = df.copy()
    
    # tenure_bucket
    def bucket_tenure(tenure):
        if tenure <= 6:
            return 0
        elif tenure <= 12:
            return 1
        elif tenure <= 24:
            return 2
        else:
            return 3
    
    df['tenure_bucket'] = df['tenure'].apply(bucket_tenure)
    
    # services_count
    df['services_count'] = (
        df['PhoneService'] + 
        (df['MultipleLines'] > 0).astype(int) +
        (df['InternetService'] < 2).astype(int) +
        (df['OnlineSecurity'] > 0).astype(int) +
        (df['OnlineBackup'] > 0).astype(int) +
        (df['DeviceProtection'] > 0).astype(int) +
        (df['TechSupport'] > 0).astype(int) +
        (df['StreamingTV'] > 0).astype(int) +
        (df['StreamingMovies'] > 0).astype(int)
    )
    
    # monthly_to_total_ratio
    expected_total = df['tenure'] * df['MonthlyCharges']
    df['monthly_to_total_ratio'] = df['TotalCharges'] / np.maximum(1, expected_total)
    
    # Flags
    df['internet_no_tech_support'] = ((df['InternetService'] < 2) & (df['TechSupport'] == 0)).astype(int)
    df['fiber_optic'] = (df['InternetService'] == 1).astype(int)
    df['month_to_month'] = (df['Contract'] == 0).astype(int)
    df['no_online_security'] = ((df['InternetService'] < 2) & (df['OnlineSecurity'] == 0)).astype(int)
    
    return df

def preprocess_input(input_data, feature_names):
    """Preprocess input for prediction"""
    df = pd.DataFrame([input_data])
    df = engineer_features_for_input(df)
    
    # Ensure all features present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    return df[feature_names]

def calculate_clv(monthly_charges, contract, tenure):
    """Calculate Customer Lifetime Value"""
    # Expected tenure based on contract
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

def get_risk_label(probability):
    """Convert probability to risk label with color"""
    if probability < 0.3:
        return "Low Risk", "low-risk"
    elif probability < 0.6:
        return "Medium Risk", "medium-risk"
    else:
        return "High Risk", "high-risk"

# ============================================================
# TAB 1: PREDICT CHURN
# ============================================================

def tab_predict():
    st.markdown('<p class="main-header">🎯 Churn Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict churn risk and estimate customer lifetime value</p>', unsafe_allow_html=True)
    
    # Load models
    models, feature_names = load_models()
    rf_explainer, xgb_explainer, logistic_importance = load_explainers()
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))
    
    with col2:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
        
        st.subheader("Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
                                     ["Bank transfer (automatic)", 
                                      "Credit card (automatic)",
                                      "Electronic check",
                                      "Mailed check"])
    
    # Encode inputs
    encoding_maps = {
        'gender': {'Female': 0, 'Male': 1},
        'SeniorCitizen': {'No': 0, 'Yes': 1},
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
    
    input_data = {
        'gender': encoding_maps['gender'][gender],
        'SeniorCitizen': encoding_maps['SeniorCitizen'][senior_citizen],
        'Partner': encoding_maps['Partner'][partner],
        'Dependents': encoding_maps['Dependents'][dependents],
        'tenure': tenure,
        'PhoneService': encoding_maps['PhoneService'][phone_service],
        'MultipleLines': encoding_maps['MultipleLines'][multiple_lines],
        'InternetService': encoding_maps['InternetService'][internet_service],
        'OnlineSecurity': encoding_maps['OnlineSecurity'][online_security],
        'OnlineBackup': encoding_maps['OnlineBackup'][online_backup],
        'DeviceProtection': encoding_maps['DeviceProtection'][device_protection],
        'TechSupport': encoding_maps['TechSupport'][tech_support],
        'StreamingTV': encoding_maps['StreamingTV'][streaming_tv],
        'StreamingMovies': encoding_maps['StreamingMovies'][streaming_movies],
        'Contract': encoding_maps['Contract'][contract],
        'PaperlessBilling': encoding_maps['PaperlessBilling'][paperless_billing],
        'PaymentMethod': encoding_maps['PaymentMethod'][payment_method],
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Predict button
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        # Preprocess
        X = preprocess_input(input_data, feature_names)
        
        # Select model
        model_choice = st.session_state.get('model_choice', 'XGBoost')
        model = models[model_choice]
        
        # Predict
        churn_prob = model.predict_proba(X)[0, 1]
        prediction = model.predict(X)[0]
        
        # Calculate CLV
        clv, expected_tenure = calculate_clv(
            monthly_charges, 
            encoding_maps['Contract'][contract],
            tenure
        )
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_label, risk_class = get_risk_label(churn_prob)
            st.markdown(f'<div class="metric-card"><h3>Churn Probability</h3><h1 class="{risk_class}">{churn_prob*100:.1f}%</h1><p>{risk_label}</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><h3>Customer Lifetime Value</h3><h1>${clv:,.0f}</h1><p>Expected Tenure: {expected_tenure} months</p></div>', unsafe_allow_html=True)
        
        with col3:
            prediction_text = "Will Churn" if prediction == 1 else "Will Stay"
            st.markdown(f'<div class="metric-card"><h3>Prediction</h3><h1>{prediction_text}</h1><p>Model: {model_choice}</p></div>', unsafe_allow_html=True)
        
        # SHAP Explanation
        st.markdown("---")
        st.subheader("🔍 What's Driving This Prediction?")
        
        # Get SHAP values
        if model_choice == 'XGBoost':
            shap_values = xgb_explainer(X)
        elif model_choice == 'Random Forest':
            shap_values = rf_explainer(X)
        else:  # Logistic Regression - use coefficient importance
            st.info("**Logistic Regression uses coefficient-based importance instead of SHAP values**")
            coef = model.coef_[0]
            feature_contrib = pd.DataFrame({
                'Feature': feature_names,
                'Value': X.values[0],
                'Contribution': coef * X.values[0]
            })
            feature_contrib['Abs Contribution'] = feature_contrib['Contribution'].abs()
            feature_contrib = feature_contrib.sort_values('Abs Contribution', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['red' if c > 0 else 'green' for c in feature_contrib['Contribution']]
            ax.barh(range(len(feature_contrib)), feature_contrib['Contribution'], color=colors, alpha=0.7)
            ax.set_yticks(range(len(feature_contrib)))
            ax.set_yticklabels(feature_contrib['Feature'])
            ax.set_xlabel('Contribution to Churn Prediction')
            ax.set_title('Top 10 Feature Contributions')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()
            
            st.caption("🔴 Red = increases churn risk | 🟢 Green = decreases churn risk")
            return
        
        # For tree models, plot SHAP waterfall
        if hasattr(shap_values, 'values'):
            vals = shap_values.values[0]
            base_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0
        else:
            vals = shap_values[0]
            base_value = 0
        
        # Get top features
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': vals,
            'Feature Value': X.values[0]
        })
        feature_importance['Abs SHAP'] = feature_importance['SHAP Value'].abs()
        feature_importance = feature_importance.sort_values('Abs SHAP', ascending=False).head(10)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if s > 0 else 'green' for s in feature_importance['SHAP Value']]
        ax.barh(range(len(feature_importance)), feature_importance['SHAP Value'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(feature_importance)))
        ax.set_yticklabels([f"{f} = {v:.2f}" for f, v in zip(feature_importance['Feature'], feature_importance['Feature Value'])])
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title(f'Top 10 Features Driving Prediction ({model_choice})')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()
        
        st.caption("🔴 Red = pushes toward churn | 🟢 Green = pushes toward retention")
        
        # CLV Formula
        st.markdown("---")
        st.subheader("💡 CLV Calculation")
        st.info(f"""
        **Customer Lifetime Value = Monthly Charges × Expected Tenure**
        
        - Monthly Charges: ${monthly_charges:.2f}
        - Current Tenure: {tenure} months
        - Contract Type: {contract}
        - Expected Tenure: {expected_tenure} months
        - **CLV = ${monthly_charges:.2f} × {expected_tenure} = ${clv:,.2f}**
        """)

# ============================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================

def tab_model_performance():
    st.markdown('<p class="main-header">📈 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Compare model metrics and understand feature importance</p>', unsafe_allow_html=True)
    
    # Load comparison
    comparison = load_model_comparison()
    
    # Model comparison table
    st.subheader("🏆 Model Comparison")
    
    # Format and display
    comparison_display = comparison.copy()
    comparison_display = comparison_display.round(4)
    
    # Highlight best values
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: lightgreen' if v else '' for v in is_max]
    
    st.dataframe(
        comparison_display.style.apply(highlight_max, axis=0)
    )
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    best_model = comparison['auc_roc'].idxmax()
    best_auc = comparison['auc_roc'].max()
    
    with col1:
        st.metric("Best Model", best_model)
    with col2:
        st.metric("Best AUC-ROC", f"{best_auc:.4f}")
    with col3:
        avg_recall = comparison['recall'].mean()
        st.metric("Average Recall", f"{avg_recall:.4f}")
    
    # ROC Curves
    st.markdown("---")
    st.subheader("📊 ROC Curves")
    
    if os.path.exists('data/processed/roc_curves.png'):
        roc_img = Image.open('data/processed/roc_curves.png')
        st.image(roc_img, use_container_width=True)
    else:
        st.warning("ROC curves plot not found")
    
    # Confusion Matrices
    st.markdown("---")
    st.subheader("🎯 Confusion Matrices")
    
    if os.path.exists('data/processed/confusion_matrices.png'):
        cm_img = Image.open('data/processed/confusion_matrices.png')
        st.image(cm_img, use_container_width=True)
    else:
        st.warning("Confusion matrices plot not found")
    
    # Feature Importance
    st.markdown("---")
    st.subheader("🔍 Feature Importance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Logistic Regression**")
        if os.path.exists('data/processed/logistic_importance.png'):
            log_img = Image.open('data/processed/logistic_importance.png')
            st.image(log_img, use_container_width=True)
    
    with col2:
        st.markdown("**Random Forest (SHAP)**")
        if os.path.exists('data/processed/shap_importance_random_forest.png'):
            rf_img = Image.open('data/processed/shap_importance_random_forest.png')
            st.image(rf_img, use_container_width=True)
    
    st.markdown("**XGBoost (SHAP)**")
    if os.path.exists('data/processed/shap_importance_xgboost.png'):
        xgb_img = Image.open('data/processed/shap_importance_xgboost.png')
        st.image(xgb_img, use_container_width=True)
    
    # Feature comparison across models
    st.markdown("---")
    st.subheader("🔬 Feature Importance Comparison")
    
    if os.path.exists('data/processed/feature_importance_comparison.csv'):
        feat_comp = pd.read_csv('data/processed/feature_importance_comparison.csv')
        st.dataframe(feat_comp.head(15))
        
        st.info("""
        **Key Insights:**
        - **Contract type** is consistently the most important feature
        - **Tenure** and **MonthlyCharges** are top predictors across all models
        - **Month-to-month** contracts and **fiber optic** internet strongly predict churn
        """)

# ============================================================
# TAB 3: CLV OVERVIEW
# ============================================================

def tab_clv_overview():
    st.markdown('<p class="main-header">💰 Customer Lifetime Value Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understand customer value and prioritize retention efforts</p>', unsafe_allow_html=True)
    
    # Load data
    train, val, test = load_processed_data()
    df = pd.concat([train, val, test], ignore_index=True)
    
    # Segment by CLV
    df['CLV_segment'] = pd.qcut(df['CLV'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])
    
    # CLV Distribution
    st.subheader("📊 CLV Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists('data/processed/clv_distribution.png'):
            clv_dist_img = Image.open('data/processed/clv_distribution.png')
            st.image(clv_dist_img, use_container_width=True)
    
    with col2:
        # Summary statistics
        st.markdown("**CLV Statistics**")
        clv_stats = df['CLV'].describe()
        st.dataframe(clv_stats.round(2))
    
    # Churn by CLV Segment
    st.markdown("---")
    st.subheader("📉 Churn Rate by CLV Segment")
    
    if os.path.exists('data/processed/churn_by_clv.png'):
        churn_clv_img = Image.open('data/processed/churn_by_clv.png')
        st.image(churn_clv_img, use_container_width=True)
    
    # Churn analysis table
    churn_by_segment = df.groupby('CLV_segment', observed=True).agg({
        'Churn': ['count', 'sum', 'mean'],
        'CLV': 'mean'
    }).round(3)
    
    churn_by_segment.columns = ['Total Customers', 'Churned', 'Churn Rate', 'Avg CLV']
    churn_by_segment['Churn Rate'] = (churn_by_segment['Churn Rate'] * 100).round(1).astype(str) + '%'
    churn_by_segment['Avg CLV'] = '$' + churn_by_segment['Avg CLV'].round(0).astype(int).astype(str)
    
    st.dataframe(churn_by_segment)
    
    # Business Insights
    st.markdown("---")
    st.subheader("💡 Business Insights & Recommendations")
    
    # Load insights
    insights_text = load_clv_insights()
    st.info(insights_text)
    
    # Actionable recommendations
    st.markdown("---")
    st.subheader("🎯 Retention Strategy Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🥇 Priority 1: Premium Segment")
        premium_churned = len(df[(df['CLV_segment'] == 'Premium') & (df['Churn'] == 1)])
        premium_revenue = df[(df['CLV_segment'] == 'Premium') & (df['Churn'] == 1)]['CLV'].sum()
        st.markdown(f"""
        - **At Risk:** {premium_churned} customers
        - **Revenue at Risk:** ${premium_revenue:,.0f}
        - **Action:** Personalized retention offers, dedicated account manager
        - **Budget:** High (max ROI)
        """)
    
    with col2:
        st.markdown("### 🥈 Priority 2: High Segment")
        high_churned = len(df[(df['CLV_segment'] == 'High') & (df['Churn'] == 1)])
        high_revenue = df[(df['CLV_segment'] == 'High') & (df['Churn'] == 1)]['CLV'].sum()
        st.markdown(f"""
        - **At Risk:** {high_churned} customers
        - **Revenue at Risk:** ${high_revenue:,.0f}
        - **Action:** Proactive engagement, service upgrades
        - **Budget:** Medium-High
        """)
    
    with col3:
        st.markdown("### 🥉 Priority 3: Medium/Low")
        medium_low_churned = len(df[(df['CLV_segment'].isin(['Medium', 'Low'])) & (df['Churn'] == 1)])
        st.markdown(f"""
        - **At Risk:** {medium_low_churned} customers
        - **Action:** Automated campaigns, improve service quality
        - **Budget:** Low (scalable interventions)
        """)
    
    # Key takeaways
    st.markdown("---")
    st.success("""
    **🎯 Key Takeaways:**
    
    1. **Focus retention efforts on Premium and High CLV segments** - they represent the majority of revenue at risk
    2. **Month-to-month contracts are the biggest churn driver** - encourage longer-term commitments
    3. **Fiber optic customers churn more** - improve service quality and support
    4. **Early intervention is key** - predict and act before customers reach high churn probability
    """)

# ============================================================
# MAIN APP
# ============================================================

def main():
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["XGBoost", "Random Forest", "Logistic Regression"],
        help="Choose which model to use for predictions"
    )
    st.session_state['model_choice'] = model_choice
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 About This App")
    st.sidebar.info("""
    This app predicts customer churn and estimates Customer Lifetime Value (CLV) for a telecom company.
    
    **Features:**
    - 3 ML models with 84%+ AUC-ROC
    - SHAP explanations
    - CLV segmentation
    - Business insights
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Resources")
    st.sidebar.markdown("[GitHub Repo](#) | [Video Demo](#)")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📈 Model Performance", "💰 CLV Overview"])
    
    with tab1:
        tab_predict()
    
    with tab2:
        tab_model_performance()
    
    with tab3:
        tab_clv_overview()

if __name__ == "__main__":
    main()