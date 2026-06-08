import os
import pandas as pd
import statsmodels.api as sm
import config
from data_preparation import load_and_merge, filter_sparse_series
from feature_engineering import engineer_features

def run_robustness_analysis():
    print("--- Starting Robustness Analysis (OLS & Oster Bounds) ---")
    
    # 1. Load Data
    if os.path.exists(config.FEATURES_PATH):
        df = pd.read_pickle(config.FEATURES_PATH)
    else:
        df = load_and_merge()
        df = filter_sparse_series(df)
        df = engineer_features(df)

    Y = df[config.Y_COL]
    D = df[config.T_COL]
    W = df[config.W_COLS + config.X_COLS] # All controls

    # --- Step 1: Short Regression (OLS without controls) ---
    print("\nFitting Short OLS Regression (Outcome ~ Treatment)...")
    X_short = sm.add_constant(D)
    res_short = sm.OLS(Y, X_short).fit()
    beta_short = res_short.params[config.T_COL]
    r2_short = res_short.rsquared
    print(f"Short OLS: Coeff({config.T_COL}) = {beta_short:.4f}, R-squared = {r2_short:.4f}")

    # --- Step 2: Long Regression (OLS with all controls) ---
    print("\nFitting Long OLS Regression (Outcome ~ Treatment + Controls)...")
    X_long = sm.add_constant(pd.concat([D, W], axis=1))
    res_long = sm.OLS(Y, X_long).fit()
    beta_long = res_long.params[config.T_COL]
    r2_long = res_long.rsquared
    print(f"Long OLS: Coeff({config.T_COL}) = {beta_long:.4f}, R-squared = {r2_long:.4f}")

    # --- Step 3: Oster (2019) Bounds Logic (Simplified) ---
    # This simplified logic checks how much the coefficient changed after adding controls
    # and compares it to the change in R-squared.
    # If the coefficient changes significantly, but R^2 doesn't increase much,
    # it suggests unobserved confounders would need to be very strong.
    # Conversely, if R^2 increases a lot and coeff stabilizes, it's good.
    
    print("\n--- Oster (2019) Bounds Logic (Simplified) ---")
    print(f"Coefficient change (Short OLS to Long OLS): {beta_short:.4f} -> {beta_long:.4f}")
    print(f"R-squared change (Short OLS to Long OLS): {r2_short:.4f} -> {r2_long:.4f}")

    # Calculate the percentage change in coefficient
    if beta_short != 0:
        coeff_change_percent = abs((beta_long - beta_short) / beta_short) * 100
    else:
        coeff_change_percent = float('inf') # Handle division by zero

    # Calculate the change in R-squared due to controls
    delta_r2 = r2_long - r2_short

    print(f"Percentage change in coefficient: {coeff_change_percent:.2f}%")
    print(f"Increase in R-squared from controls: {delta_r2:.4f}")

    # Interpretation based on Oster's logic (simplified)
    # If delta_r2 is small, but coeff_change_percent is large, it's concerning.
    # If delta_r2 is large, and coeff_change_percent is large, it means controls are important.
    # If delta_r2 is large, and coeff_change_percent is small, it means controls are important but don't change the core effect much.
    
    if delta_r2 < config.OSTER_DELTA_R2_THRESHOLD and coeff_change_percent > 50: # Example thresholds
        print("Interpretation: Adding controls had a limited impact on R-squared, but a large impact on the coefficient.")
        print("This suggests that unobserved confounders would need to be extremely powerful to explain away the remaining effect.")
    elif delta_r2 >= config.OSTER_DELTA_R2_THRESHOLD and coeff_change_percent > 50:
        print("Interpretation: Controls significantly improved model fit and changed the coefficient substantially.")
        print("This indicates the importance of controlling for observed confounders.")
    else:
        print("Interpretation: The coefficient appears relatively robust to observed confounding, or controls are effective.")

if __name__ == "__main__":
    run_robustness_analysis()