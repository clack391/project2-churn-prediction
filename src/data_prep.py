"""
Data Preparation for Customer Churn Prediction
Handles feature engineering, encoding, and train/val/test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_data(url=None):
    """Load the IBM Telco Customer Churn dataset"""
    if url is None:
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    print("Loading data from URL...")
    df = pd.read_csv(url)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_data(df):
    """Clean the dataset and handle missing values"""
    df = df.copy()
    
    # Handle TotalCharges missing values
    # Approach: TotalCharges has whitespace for new customers (tenure=0)
    # We'll convert to numeric and fill with 0 for tenure=0 customers
    print("\nHandling missing values in TotalCharges...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges: if tenure is 0, charge should be 0
    missing_mask = df['TotalCharges'].isna()
    print(f"Found {missing_mask.sum()} missing TotalCharges values")
    
    df.loc[missing_mask & (df['tenure'] == 0), 'TotalCharges'] = 0
    df.loc[missing_mask & (df['tenure'] > 0), 'TotalCharges'] = df['MonthlyCharges']
    
    print(f"After handling: {df['TotalCharges'].isna().sum()} missing values remain")
    
    return df

def engineer_features(df):
    """Create business-driven features"""
    df = df.copy()
    
    print("\nEngineering features...")
    
    # 1. tenure_bucket: 0-6m, 6-12m, 12-24m, 24m+
    def bucket_tenure(tenure):
        if tenure <= 6:
            return '0-6m'
        elif tenure <= 12:
            return '6-12m'
        elif tenure <= 24:
            return '12-24m'
        else:
            return '24m+'
    
    df['tenure_bucket'] = df['tenure'].apply(bucket_tenure)
    
    # 2. services_count: total number of services
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Count services (excluding 'No' and 'No internet service' and 'No phone service')
    df['services_count'] = 0
    for col in service_cols:
        if col in df.columns:
            df['services_count'] += (
                (df[col] == 'Yes') | 
                ((df[col] != 'No') & (df[col] != 'No internet service') & (df[col] != 'No phone service'))
            ).astype(int)
    
    # 3. monthly_to_total_ratio: TotalCharges / max(1, tenure * MonthlyCharges)
    expected_total = df['tenure'] * df['MonthlyCharges']
    df['monthly_to_total_ratio'] = df['TotalCharges'] / np.maximum(1, expected_total)
    
    # 4. Flag: internet but no tech support
    df['internet_no_tech_support'] = (
        (df['InternetService'].isin(['DSL', 'Fiber optic'])) & 
        (df['TechSupport'] == 'No')
    ).astype(int)
    
    # Additional useful flags
    df['fiber_optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['no_online_security'] = (
        (df['InternetService'].isin(['DSL', 'Fiber optic'])) & 
        (df['OnlineSecurity'] == 'No')
    ).astype(int)
    
    print(f"Created features: tenure_bucket, services_count, monthly_to_total_ratio, and flags")
    
    return df

def calculate_expected_tenure(df):
    """
    Calculate Expected Tenure based on contract type and current tenure
    
    Assumption:
    - Month-to-month: Expected to stay 6 more months on average
    - One year: Expected to complete contract + 6 months
    - Two year: Expected to complete contract + 12 months
    """
    df = df.copy()
    
    print("\nCalculating Expected Tenure...")
    print("Assumption: Contract renewals + historical average remaining tenure")
    
    def get_expected_tenure(row):
        current_tenure = row['tenure']
        contract = row['Contract']
        
        if contract == 'Month-to-month':
            # Average 6 more months
            return current_tenure + 6
        elif contract == 'One year':
            # Complete current year + 6 months
            months_in_current_contract = current_tenure % 12
            months_to_complete = 12 - months_in_current_contract if months_in_current_contract > 0 else 0
            return current_tenure + months_to_complete + 6
        else:  # Two year
            # Complete current contract + 12 months
            months_in_current_contract = current_tenure % 24
            months_to_complete = 24 - months_in_current_contract if months_in_current_contract > 0 else 0
            return current_tenure + months_to_complete + 12
    
    df['expected_tenure'] = df.apply(get_expected_tenure, axis=1)
    
    print(f"Expected tenure calculated (mean: {df['expected_tenure'].mean():.1f} months)")
    
    return df

def calculate_clv(df):
    """Calculate Customer Lifetime Value"""
    df = df.copy()
    
    print("\nCalculating CLV...")
    # CLV = MonthlyCharges × ExpectedTenure
    df['CLV'] = df['MonthlyCharges'] * df['expected_tenure']
    
    print(f"CLV calculated (mean: ${df['CLV'].mean():.2f}, median: ${df['CLV'].median():.2f})")
    
    return df

def encode_categorical(df):
    """
    Encode categorical variables using LabelEncoder
    IMPORTANT: LabelEncoder sorts alphabetically!
    """
    df = df.copy()
    
    print("\nEncoding categorical variables...")
    
    # Columns to encode
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'tenure_bucket'
    ]
    
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            
            # Print first few mappings for verification
            if col in ['gender', 'MultipleLines', 'Contract']:
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                print(f"  {col}: {mapping}")
    
    # Encode target variable
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    print(f"Encoded {len(categorical_cols)} categorical columns")
    
    return df, label_encoders

def select_features(df):
    """Select features for modeling"""
    # Drop identifier and target
    feature_cols = [col for col in df.columns if col not in ['customerID', 'Churn', 'CLV', 'expected_tenure']]
    
    X = df[feature_cols]
    y = df['Churn']
    
    print(f"\nSelected {len(feature_cols)} features for modeling")
    
    return X, y, feature_cols

def split_data(df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train/val/test with stratification
    60% train, 20% val, 20% test
    """
    print("\nSplitting data (60% train / 20% val / 20% test)...")
    
    # First split: separate test set
    train_val, test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['Churn']
    )
    
    # Second split: separate validation from train
    val_size_adjusted = val_size / (1 - test_size)  # Adjust proportion
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val['Churn']
    )
    
    print(f"Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val)} ({len(val)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")
    
    # Check churn distribution
    print("\nChurn distribution:")
    print(f"Train: {train['Churn'].mean()*100:.1f}%")
    print(f"Val:   {val['Churn'].mean()*100:.1f}%")
    print(f"Test:  {test['Churn'].mean()*100:.1f}%")
    
    return train, val, test

def save_processed_data(train, val, test, output_dir='data/processed'):
    """Save processed datasets"""
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    
    print(f"\nProcessed data saved to {output_dir}/")
    print(f"  - train.csv: {len(train)} rows")
    print(f"  - val.csv: {len(val)} rows")
    print(f"  - test.csv: {len(test)} rows")

def prepare_data_pipeline():
    """Main pipeline to prepare data"""
    print("="*60)
    print("CUSTOMER CHURN DATA PREPARATION PIPELINE")
    print("="*60)
    
    # 1. Load data
    df = load_data()
    
    # 2. Clean data
    df = clean_data(df)
    
    # 3. Engineer features
    df = engineer_features(df)
    
    # 4. Calculate expected tenure and CLV
    df = calculate_expected_tenure(df)
    df = calculate_clv(df)
    
    # 5. Encode categorical variables
    df, label_encoders = encode_categorical(df)
    
    # 6. Split data
    train, val, test = split_data(df)
    
    # 7. Save processed data
    save_processed_data(train, val, test)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    
    return train, val, test, label_encoders

if __name__ == "__main__":
    # Run the pipeline
    train, val, test, label_encoders = prepare_data_pipeline()
    
    # Display sample
    print("\nSample of processed data:")
    print(train.head())
    
    print("\nFeature names:")
    feature_cols = [col for col in train.columns if col not in ['customerID', 'Churn']]
    print(feature_cols)