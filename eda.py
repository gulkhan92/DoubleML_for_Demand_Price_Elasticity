import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config
from data_preparation import load_and_merge, filter_sparse_series
from feature_engineering import engineer_features

def run_eda():
    """
    Performs Exploratory Data Analysis (EDA) covering basic data science metrics
    and project-specific causal analysis for price elasticity.
    """
    print("--- Starting Exploratory Data Analysis ---")

    # Create directory for plots
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # 1. Load Data
    if os.path.exists(config.FEATURES_PATH):
        print(f"Loading cached features from {config.FEATURES_PATH}...")
        df = pd.read_pickle(config.FEATURES_PATH)
    else:
        print("Processing raw data for EDA...")
        df = load_and_merge()
        df = filter_sparse_series(df)
        df = engineer_features(df)

    # --- [Basic EDA: Data Science & ML] ---
    print("\n[Basic EDA] Descriptive Statistics:")
    print(df.info())
    print(df[[config.Y_COL, config.T_COL, 'sales', 'sell_price']].describe())

    # Distribution of Unit Sales
    plt.figure(figsize=(12, 6))
    sns.histplot(df['sales'], bins=50, kde=True, log_scale=True)
    plt.title('Distribution of Sales Volume (Log Scale)')
    plt.xlabel('Sales')
    plt.savefig(os.path.join(plot_dir, 'sales_distribution.png'))
    plt.close()

    # Time Series trend
    plt.figure(figsize=(14, 6))
    df_agg = df.groupby('date')['sales'].sum().reset_index()
    df_agg['date'] = pd.to_datetime(df_agg['date'])
    sns.lineplot(data=df_agg, x='date', y='sales')
    plt.title('Total Weekly Sales Volume Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'weekly_sales_trend.png'))
    plt.close()

    # --- [Project Specific EDA: Demand Price Elasticity] ---
    print("\n[Project Specific EDA] Causal Diagnostics: Elasticity and Confounding:")

    # Scatter plot with regression line (Log-Log)
    # Sampling for efficient plotting of large dataset
    sample_df = df.sample(n=min(100000, len(df)), random_state=42)
    plt.figure(figsize=(10, 6))
    sns.regplot(data=sample_df, x=config.T_COL, y=config.Y_COL, 
                scatter_kws={'alpha': 0.1, 's': 1}, line_kws={'color': 'red'})
    plt.title(f'Naive Log-Log Correlation (Elasticity Baseline)')
    plt.xlabel('Log(Price)')
    plt.ylabel('Log(Quantity)')
    plt.savefig(os.path.join(plot_dir, 'price_quantity_correlation.png'))
    plt.close()

    # Price Volatility (Coefficient of Variation)
    print("\nPrice Volatility (Coefficient of Variation) per Item-Store:")
    price_cv = df.groupby(['item_id', 'store_id'])['sell_price'].apply(lambda x: np.std(x) / np.mean(x) if np.mean(x) > 0 else 0)
    print(f"Mean Price CV: {price_cv.mean():.4f}")
    print(f"Median Price CV: {price_cv.median():.4f}")
    print(f"Items with zero price variation: {(price_cv == 0).sum()} out of {len(price_cv)}")

    plt.figure(figsize=(10, 6))
    sns.histplot(price_cv[price_cv > 0], bins=50, kde=True)
    plt.title('Distribution of Price Coefficient of Variation (CV)')
    plt.xlabel('Price CV')
    plt.savefig(os.path.join(plot_dir, 'price_cv_distribution.png'))
    plt.close()

    # Log-Price vs Lagged Sales (Endogeneity Check)
    # Sample for performance
    sample_df_lag = df.sample(n=min(100000, len(df)), random_state=43)
    plt.figure(figsize=(10, 6))
    sns.regplot(data=sample_df_lag, x='lag_sales_1', y=config.T_COL, 
                scatter_kws={'alpha': 0.1, 's': 1}, line_kws={'color': 'red'})
    plt.title('Log-Price vs Lagged Sales (Potential Endogeneity)')
    plt.xlabel('Lagged Log(Quantity + 1)')
    plt.ylabel('Log(Price)')
    plt.savefig(os.path.join(plot_dir, 'price_lag_sales_correlation.png'))
    plt.close()

    # Correlation Heatmap for Confounders
    plt.figure(figsize=(12, 10))
    # Select subset of columns for heatmap
    cols_for_heatmap = [config.Y_COL, config.T_COL, 'month', 'snap_CA', 'snap_TX', 'snap_WI', 'lag_sales_1', 'lag_sales_2']
    # Filter only numeric/encoded columns
    available_cols = [c for c in cols_for_heatmap if c in df.columns]
    corr = df[available_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Correlation Heatmap (Treatment, Outcome, and Confounders)')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'correlation_heatmap.png'))
    plt.close()

    # Distribution of Log-Quantity by Department (Hint for Heterogeneity)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df.sample(n=min(200000, len(df)), random_state=44), x='dept_id', y=config.Y_COL)
    plt.title('Distribution of Log-Quantity by Department')
    plt.savefig(os.path.join(plot_dir, 'log_quantity_by_department.png'))
    plt.close()

    print(f"\nEDA Complete. Visuals saved to '{plot_dir}/' directory.")

if __name__ == "__main__":
    run_eda()