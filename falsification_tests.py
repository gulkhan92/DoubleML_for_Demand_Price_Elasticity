import os
import pandas as pd
import numpy as np
import config
from model_training import run_double_ml
from data_preparation import load_and_merge, filter_sparse_series
from feature_engineering import engineer_features

def run_falsification_tests():
    print("--- Starting Falsification Tests ---")
    
    # 1. Load Data
    if os.path.exists(config.FEATURES_PATH):
        df = pd.read_pickle(config.FEATURES_PATH)
    else:
        df = load_and_merge()
        df = filter_sparse_series(df)
        df = engineer_features(df)

    # --- Test 1: Placebo Treatment (Shuffled Price) ---
    print("\n--- Running Placebo Test: Shuffling Treatment (log_price) ---")
    
    # Create a copy to avoid modifying the original DataFrame for subsequent tests
    df_placebo = df.copy()
    
    # Randomly shuffle the treatment variable (log_price)
    # This breaks any true causal link between price and quantity
    df_placebo[config.PLACEBO_SHUFFLE_COL] = df_placebo[config.PLACEBO_SHUFFLE_COL].sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    print(f"Running DML with shuffled '{config.PLACEBO_SHUFFLE_COL}'...")
    try:
        dml_placebo = run_double_ml(df_placebo, learner_type="lightgbm") # Use LightGBM for speed
        placebo_coeff = dml_placebo.coef[0]
        placebo_pval = dml_placebo.pval[0]
        
        print(f"Placebo Test Result: Elasticity = {placebo_coeff:.4f}, p-value = {placebo_pval:.4f}")
        if abs(placebo_coeff) < 0.01 and placebo_pval > 0.05: # Example thresholds
            print("Interpretation: SUCCESS. Estimated elasticity is close to zero and not statistically significant.")
            print("This suggests the DML method is not finding spurious effects when the true link is broken.")
        else:
            print("Interpretation: WARNING. Estimated elasticity is significant or not close to zero.")
            print("This might indicate issues with the DML setup picking up spurious correlations.")
    except Exception as e:
        print(f"Placebo test failed: {e}")

    # --- Test 2: Wrong Outcome (Random Noise) ---
    print("\n--- Running Wrong Outcome Test: Using Random Noise as Outcome ---")
    
    df_wrong_outcome = df.copy()
    # Create a column of random noise as the outcome variable
    df_wrong_outcome['random_outcome'] = np.random.rand(len(df_wrong_outcome))
    
    # Temporarily change Y_COL in config for this test
    original_y_col = config.Y_COL
    config.Y_COL = 'random_outcome'
    
    print(f"Running DML with '{config.T_COL}' as treatment and random noise as outcome...")
    try:
        dml_wrong_outcome = run_double_ml(df_wrong_outcome, learner_type="lightgbm")
        wrong_outcome_coeff = dml_wrong_outcome.coef[0]
        wrong_outcome_pval = dml_wrong_outcome.pval[0]
        
        print(f"Wrong Outcome Test Result: Elasticity = {wrong_outcome_coeff:.4f}, p-value = {wrong_outcome_pval:.4f}")
        if abs(wrong_outcome_coeff) < 0.01 and wrong_outcome_pval > 0.05:
            print("Interpretation: SUCCESS. Estimated elasticity is close to zero and not statistically significant.")
            print("This suggests the DML model is not finding spurious causal effects with an unrelated outcome.")
        else:
            print("Interpretation: WARNING. Estimated elasticity is significant or not close to zero.")
            print("This might indicate issues with the DML setup picking up spurious correlations.")
    except Exception as e:
        print(f"Wrong outcome test failed: {e}")
    finally:
        config.Y_COL = original_y_col # Revert Y_COL

if __name__ == "__main__":
    run_falsification_tests()