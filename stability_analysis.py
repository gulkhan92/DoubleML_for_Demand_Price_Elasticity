import os
import pandas as pd
import numpy as np
import config
from model_training import run_double_ml
from feature_engineering import engineer_features
from data_preparation import load_and_merge, filter_sparse_series

def run_stability_check():
    print("--- Starting Standard Error Stability Analysis ---")
    
    # Load data
    if os.path.exists(config.FEATURES_PATH):
        df = pd.read_pickle(config.FEATURES_PATH)
    else:
        df = load_and_merge()
        df = filter_sparse_series(df)
        df = engineer_features(df)

    coefficients = []
    std_errors = []

    print(f"Running {config.STABILITY_SAMPLES} iterations with sample size {config.SUB_SAMPLE_SIZE}...")
    
    for i in range(config.STABILITY_SAMPLES):
        # Sub-sample the data
        sub_df = df.sample(n=min(len(df), config.SUB_SAMPLE_SIZE), random_state=i)
        
        # Use LightGBM as the baseline for stability check
        dml_model = run_double_ml(sub_df, learner_type="lightgbm")
        
        coeff = dml_model.coef[0]
        se = dml_model.se[0]
        
        coefficients.append(coeff)
        std_errors.append(se)
        print(f"Iteration {i+1}: Coeff = {coeff:.4f}, SE = {se:.4f}")

    print("\n--- Stability Results ---")
    print(f"Mean Coefficient: {np.mean(coefficients):.4f}")
    print(f"Coefficient Std Dev: {np.std(coefficients):.4f}")
    print(f"Max Deviation: {max(coefficients) - min(coefficients):.4f}")
    
    if np.std(coefficients) < 0.01:
        print("Result: High Stability. The estimate is robust to data sampling.")
    else:
        print("Result: Low Stability. Consider increasing sample size or checking for outliers.")

if __name__ == "__main__":
    run_stability_check()