import statsmodels.api as sm
import config

def compare_with_ols(df):
    print("\n--- Naive OLS Results ---")
    X = sm.add_constant(df[config.T_COL])
    model = sm.OLS(df[config.Y_COL], X).fit()
    print(model.summary().tables[1])
    return model

def print_dml_summary(dml_model):
    print("\n--- Double ML Elasticity Results ---")
    print(dml_model.summary)
