import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import config
from data_preparation import load_and_merge, filter_sparse_series
from feature_engineering import engineer_features

def run_hausman_test():
    print("--- Starting Hausman Test for Price Endogeneity ---")
    
    # 1. Load Data
    if os.path.exists(config.FEATURES_PATH):
        df = pd.read_pickle(config.FEATURES_PATH)
    else:
        df = load_and_merge()
        df = filter_sparse_series(df)
        df = engineer_features(df)

    # Define variable sets
    Y = df[config.Y_COL]
    D = df[config.T_COL]
    # Controls (W) excluding the instruments
    W = df[config.EXCL_W_COLS + config.X_COLS]
    # Instruments (Z)
    Z = df[config.IV_COLS]

    # --- Step 1: Naive OLS (Baseline) ---
    print("\nFitting Naive OLS...")
    X_ols = sm.add_constant(pd.concat([D, W], axis=1))
    res_ols = sm.OLS(Y, X_ols).fit()
    beta_ols = res_ols.params[config.T_COL]

    # --- Step 2: Durbin-Wu-Hausman (Regression-based) ---
    # The logic: If price is endogenous, it is correlated with the error term.
    # We regress price on instruments and controls, get residuals, and add them to OLS.
    print("Performing Durbin-Wu-Hausman residual check...")
    X_first = sm.add_constant(pd.concat([W, Z], axis=1))
    first_stage = sm.OLS(D, X_first).fit()
    price_residuals = first_stage.resid

    # Second stage: Y ~ Price + Controls + Price_Residuals
    X_hausman = sm.add_constant(pd.concat([D, W, price_residuals.rename('price_resid')], axis=1))
    res_hausman = sm.OLS(Y, X_hausman).fit()
    
    # The p-value of the 'price_resid' coefficient tests the null of exogeneity
    hausman_p = res_hausman.pvalues['price_resid']

    # --- Step 3: 2SLS (Consistent Estimator) ---
    print("Fitting 2SLS (Instrumental Variables)...")
    # IV2SLS(endog, exog, instrument)
    # exog here must include the constant and the exogenous controls
    X_iv_second = sm.add_constant(W) 
    res_iv = IV2SLS(Y, pd.concat([D, X_iv_second], axis=1), pd.concat([Z, X_iv_second], axis=1)).fit()
    beta_iv = res_iv.params[config.T_COL]

    # --- Summary ---
    print("\n" + "="*50)
    print(f"OLS Elasticity:  {beta_ols:.4f}")
    print(f"2SLS Elasticity: {beta_iv:.4f}")
    print("-" * 50)
    print(f"Hausman Test (Residual Check) p-value: {hausman_p:.4e}")
    
    if hausman_p < 0.05:
        print("RESULT: Reject Null Hypothesis.")
        print("Price is ENDOGENOUS. OLS is biased. Trust 2SLS or DML results.")
    else:
        print("RESULT: Fail to Reject Null.")
        print("No strong evidence of endogeneity. OLS may be consistent.")
    print("="*50)

    # Note on DML agreement
    print(f"\nContextual Check: Your DML results (~ -0.09) should align closer to 2SLS ({beta_iv:.4f}) than OLS ({beta_ols:.4f}).")

if __name__ == "__main__":
    run_hausman_test()