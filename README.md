# Customer Churn Prediction & CLV Analysis

🎯 **Business Problem**: SaaS companies lose 5-7% of revenue annually to customer churn. This project predicts which customers are likely to churn and estimates their Customer Lifetime Value (CLV) to prioritize retention efforts effectively.

## 🚀 Live Demo
**App URL**: [https://clack391-project2-churn-prediction-app-yx7okn.streamlit.app/]  
**Video Demo**: [My YouTube/Loom Link]

---

## 📊 Project Overview

This end-to-end machine learning system helps businesses:
- **Predict churn risk** for individual customers with 85%+ accuracy
- **Estimate Customer Lifetime Value** to identify high-value accounts
- **Explain predictions** using SHAP values for actionable insights
- **Prioritize retention** efforts based on value and risk

### Key Features
- 3 ML models: Logistic Regression, Random Forest, XGBoost
- Interactive Streamlit app with real-time predictions
- SHAP interpretability for model transparency
- CLV segmentation and churn analysis

---

## 💰 CLV Calculation & Assumptions

### Formula
```
CLV = Monthly Charges × Expected Tenure (months)
```

### Expected Tenure Assumptions
We estimate how long a customer will remain based on their contract type:

| Contract Type | Expected Tenure Calculation |
|---------------|---------------------------|
| **Month-to-month** | Current tenure + 6 months (average renewal period) |
| **One year** | Current tenure + months to complete contract + 6 months buffer |
| **Two year** | Current tenure + months to complete contract + 12 months buffer |

**Rationale**: 
- Long-term contracts signal commitment and show lower churn rates
- Historical data shows customers typically renew for 6-12 months beyond their initial contract
- Month-to-month customers are more volatile but may stay ~6 months on average

### CLV Segments
Customers are divided into quartiles:
- **Low**: Bottom 25% of CLV
- **Medium**: 25-50th percentile
- **High**: 50-75th percentile  
- **Premium**: Top 25% of CLV

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/project2-churn-prediction.git
cd project2-churn-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 Data Preparation

### Step 1: Run Data Preparation
```bash
python src/data_prep.py
```

**What it does**:
- Downloads IBM Telco Customer Churn dataset
- Handles missing values in `TotalCharges`
- Engineers features: `tenure_bucket`, `services_count`, `monthly_to_total_ratio`, etc.
- Calculates Expected Tenure and CLV
- Encodes categorical variables (LabelEncoder, alphabetically sorted)
- Splits data: 60% train / 20% val / 20% test (stratified)
- Saves processed data to `data/processed/`

### Step 2: Run CLV Analysis
```bash
python src/clv_analysis.py
```

**What it does**:
- Segments customers into CLV quartiles
- Analyzes churn rates by segment
- Generates visualizations
- Produces business insights

**Outputs**:
- `data/processed/clv_distribution.png`
- `data/processed/churn_by_clv.png`
- `data/processed/clv_insights.txt`

---

## 🤖 Model Training

```bash
python src/train_models.py
```

Trains three models with light hyperparameter tuning:
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble decision trees
- **XGBoost**: Gradient boosting (best performance)

**Expected Performance**:
- AUC-ROC: 84-86%
- Recall: 60%+
- Models handle class imbalance using `class_weight` or `scale_pos_weight`

---

## 🖥️ Running the App

### Local Development
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Deployment
The app is deployed on **Streamlit Community Cloud**.

**Deployment steps**:
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set Python version to 3.9+
4. Deploy!

---

## 📱 App Features

### Tab 1: Predict Churn
- Input customer details (tenure, contract, services, etc.)
- Get churn probability (0-100%) with risk label
- View SHAP explanation for the prediction
- See estimated CLV

### Tab 2: Model Performance
- Compare all 3 models (Precision, Recall, F1, AUC)
- ROC curves overlay
- Confusion matrix for selected model
- Global feature importance (SHAP summary plot)

### Tab 3: CLV Overview
- CLV distribution histogram
- Churn rate by CLV segment
- Business insights and recommendations

---

## 📈 Key Findings

### CLV Insights
1. **Inverse Relationship**: Low-value customers churn at ~40% while Premium customers churn at ~15%
2. **Revenue at Risk**: Premium segment represents the highest absolute revenue loss
3. **Strategic Focus**: Retaining one Premium customer = retaining 5 Low-value customers

### Model Performance
- **Best Model**: XGBoost (AUC: 85.3%)
- **Top Features**: Contract type, Tenure, Monthly Charges
- **Interpretability**: SHAP values reveal non-linear interactions

---

## 🧪 Testing the Model

### High-Risk Profile (should predict >60% churn):
- Senior citizen: Yes
- Contract: Month-to-month
- Internet: Fiber optic
- No security/backup/tech support
- Payment: Electronic check
- Monthly charges: $100+

### Low-Risk Profile (should predict <20% churn):
- Contract: Two year
- Tenure: 24+ months
- Internet: DSL
- Multiple services active
- Automatic payment method

---

## 📂 Project Structure

```
project2-churn-prediction/
├── README.md                    # This file
├── AI_USAGE.md                  # AI assistance documentation
├── requirements.txt             # Python dependencies
├── app.py                       # Streamlit app
├── data/
│   ├── raw/                     # Original dataset
│   └── processed/               # Train/val/test splits + plots
├── src/
│   ├── data_prep.py            # Data preparation pipeline
│   ├── clv_analysis.py         # CLV segmentation & insights
│   ├── train_models.py         # Model training & evaluation
│   ├── interpretability.py     # SHAP analysis
│   └── predict.py              # Prediction utilities
├── models/
│   ├── logistic.pkl            # Trained Logistic Regression
│   ├── rf.pkl                  # Trained Random Forest
│   ├── xgb.pkl                 # Trained XGBoost
│   └── preprocessor.pkl        # Feature encoder/scaler
└── notebooks/
    └── exploration.ipynb       # EDA (optional)
```

---

## 🎯 Business Value

### Who to Retain First?
**Priority 1: High & Premium CLV customers with churn risk >40%**
- Highest revenue impact
- Personalized retention offers (e.g., discounts, dedicated support)

**Priority 2: Medium CLV customers showing early warning signs**
- Preventive engagement before they reach high risk
- Cost-effective interventions

**Priority 3: Low CLV customers**
- Automated retention campaigns
- Focus on service improvements rather than individual outreach

---

## 🤝 Contributing

This is a learning project. Feedback and suggestions are welcome!

---

## 📄 License

This project is for educational purposes as part of Pioneer Academy's Data Science curriculum.

---

## 🙏 Acknowledgments

- **Dataset**: IBM Telco Customer Churn dataset
- **Course**: Pioneer Academy Data Science Project 2
- **Tools**: Streamlit, scikit-learn, XGBoost, SHAP

---

## 📧 Contact

[Precious Onotu]  
[clack391@gmail.com]  
[https://www.linkedin.com/in/precious-onotu-aba9b6312]  
[https://github.com/clack391/project2-churn-prediction]