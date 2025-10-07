"""
Customer Lifetime Value (CLV) Analysis
Segments customers by CLV and analyzes churn patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_processed_data(data_dir='data/processed'):
    """Load processed datasets"""
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Combine for full analysis
    df = pd.concat([train, val, test], ignore_index=True)
    
    print(f"Loaded {len(df)} total customers for CLV analysis")
    return df

def segment_by_clv(df):
    """
    Segment customers into CLV quartiles
    Returns df with CLV_segment column
    """
    df = df.copy()
    
    # Create quartiles
    df['CLV_segment'] = pd.qcut(
        df['CLV'], 
        q=4, 
        labels=['Low', 'Medium', 'High', 'Premium']
    )
    
    print("\nCLV Segmentation:")
    print(df.groupby('CLV_segment')['CLV'].agg(['count', 'min', 'max', 'mean']))
    
    return df

def analyze_churn_by_clv(df):
    """Analyze churn rates across CLV segments"""
    print("\n" + "="*60)
    print("CHURN RATE BY CLV SEGMENT")
    print("="*60)
    
    churn_by_segment = df.groupby('CLV_segment').agg({
        'Churn': ['count', 'sum', 'mean'],
        'CLV': 'mean'
    }).round(3)
    
    churn_by_segment.columns = ['Total_Customers', 'Churned', 'Churn_Rate', 'Avg_CLV']
    churn_by_segment['Churn_Rate_Pct'] = (churn_by_segment['Churn_Rate'] * 100).round(1)
    
    print(churn_by_segment)
    
    return churn_by_segment

def plot_clv_distribution(df, output_dir='data/processed'):
    """Plot CLV distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['CLV'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Customer Lifetime Value ($)', fontsize=12)
    axes[0].set_ylabel('Number of Customers', fontsize=12)
    axes[0].set_title('CLV Distribution', fontsize=14, fontweight='bold')
    axes[0].axvline(df['CLV'].mean(), color='red', linestyle='--', label=f'Mean: ${df["CLV"].mean():.0f}')
    axes[0].axvline(df['CLV'].median(), color='green', linestyle='--', label=f'Median: ${df["CLV"].median():.0f}')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot by segment
    segment_order = ['Low', 'Medium', 'High', 'Premium']
    df_plot = df.copy()
    df_plot['CLV_segment'] = pd.Categorical(df_plot['CLV_segment'], categories=segment_order, ordered=True)
    
    sns.boxplot(data=df_plot, x='CLV_segment', y='CLV', ax=axes[1], palette='Set2')
    axes[1].set_xlabel('CLV Segment', fontsize=12)
    axes[1].set_ylabel('Customer Lifetime Value ($)', fontsize=12)
    axes[1].set_title('CLV by Segment', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'clv_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/clv_distribution.png")
    plt.close()

def plot_churn_by_clv(df, output_dir='data/processed'):
    """Plot churn rate by CLV segment"""
    churn_summary = df.groupby('CLV_segment')['Churn'].agg(['mean', 'count']).reset_index()
    churn_summary.columns = ['CLV_segment', 'churn_rate', 'count']
    churn_summary['churn_rate_pct'] = churn_summary['churn_rate'] * 100
    
    # Ensure correct order
    segment_order = ['Low', 'Medium', 'High', 'Premium']
    churn_summary['CLV_segment'] = pd.Categorical(
        churn_summary['CLV_segment'], 
        categories=segment_order, 
        ordered=True
    )
    churn_summary = churn_summary.sort_values('CLV_segment')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of churn rate
    bars = axes[0].bar(
        churn_summary['CLV_segment'], 
        churn_summary['churn_rate_pct'],
        color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'],
        edgecolor='black',
        alpha=0.8
    )
    
    axes[0].set_xlabel('CLV Segment', fontsize=12)
    axes[0].set_ylabel('Churn Rate (%)', fontsize=12)
    axes[0].set_title('Churn Rate by CLV Segment', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width()/2., 
            height,
            f'{height:.1f}%',
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    # Customer count by segment with churn split
    churned = df[df['Churn'] == 1].groupby('CLV_segment').size()
    retained = df[df['Churn'] == 0].groupby('CLV_segment').size()
    
    x = np.arange(len(segment_order))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, retained.reindex(segment_order, fill_value=0), 
                        width, label='Retained', color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = axes[1].bar(x + width/2, churned.reindex(segment_order, fill_value=0), 
                        width, label='Churned', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    axes[1].set_xlabel('CLV Segment', fontsize=12)
    axes[1].set_ylabel('Number of Customers', fontsize=12)
    axes[1].set_title('Customer Retention by CLV Segment', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(segment_order)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'churn_by_clv.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/churn_by_clv.png")
    plt.close()

def generate_business_insights(df):
    """Generate key business insights from CLV analysis"""
    print("\n" + "="*60)
    print("BUSINESS INSIGHTS")
    print("="*60)
    
    # Calculate key metrics
    churn_by_segment = df.groupby('CLV_segment').agg({
        'Churn': 'mean',
        'CLV': 'mean'
    }).round(3)
    
    low_churn = churn_by_segment.loc['Low', 'Churn'] * 100
    premium_churn = churn_by_segment.loc['Premium', 'Churn'] * 100
    
    low_clv = churn_by_segment.loc['Low', 'CLV']
    premium_clv = churn_by_segment.loc['Premium', 'CLV']
    
    premium_count = len(df[df['CLV_segment'] == 'Premium'])
    premium_churned = len(df[(df['CLV_segment'] == 'Premium') & (df['Churn'] == 1)])
    premium_revenue_at_risk = df[(df['CLV_segment'] == 'Premium') & (df['Churn'] == 1)]['CLV'].sum()
    
    insights = []
    
    # Insight 1: Inverse relationship
    insight1 = (
        f"📊 **Inverse CLV-Churn Relationship**: Low-value customers churn at {low_churn:.1f}% "
        f"while Premium customers churn at only {premium_churn:.1f}%. "
        f"High-value customers show stronger loyalty."
    )
    insights.append(insight1)
    print(f"\n1. {insight1}")
    
    # Insight 2: Revenue at risk
    insight2 = (
        f"💰 **Premium Revenue at Risk**: {premium_churned} Premium customers ({premium_churn:.1f}%) "
        f"represent ${premium_revenue_at_risk:,.0f} in lifetime value at risk. "
        f"Retention efforts should prioritize this segment for maximum ROI."
    )
    insights.append(insight2)
    print(f"\n2. {insight2}")
    
    # Insight 3: Strategic focus
    value_ratio = premium_clv / low_clv
    insight3 = (
        f"🎯 **Strategic Focus**: Premium customers are worth {value_ratio:.1f}x more than Low-value customers "
        f"(${premium_clv:,.0f} vs ${low_clv:,.0f}). "
        f"Even small improvements in Premium retention yield significant returns."
    )
    insights.append(insight3)
    print(f"\n3. {insight3}")
    
    print("\n" + "="*60)
    
    return insights

def run_clv_analysis():
    """Main CLV analysis pipeline"""
    print("="*60)
    print("CUSTOMER LIFETIME VALUE ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_processed_data()
    
    # Segment by CLV
    df = segment_by_clv(df)
    
    # Analyze churn by CLV
    churn_summary = analyze_churn_by_clv(df)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_clv_distribution(df)
    plot_churn_by_clv(df)
    
    # Generate insights
    insights = generate_business_insights(df)
    
    # Save insights to file
    insights_path = 'data/processed/clv_insights.txt'
    with open(insights_path, 'w') as f:
        f.write("BUSINESS INSIGHTS FROM CLV ANALYSIS\n")
        f.write("="*60 + "\n\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n\n")
    
    print(f"\nInsights saved to: {insights_path}")
    
    print("\n" + "="*60)
    print("CLV ANALYSIS COMPLETE!")
    print("="*60)
    
    return df, churn_summary, insights

if __name__ == "__main__":
    df, churn_summary, insights = run_clv_analysis()