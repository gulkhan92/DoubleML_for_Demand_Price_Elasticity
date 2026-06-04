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

def compare_dml_results(results_dict):
    print("\n--- DML Model Comparison (Learner Robustness) ---")
    print(f"{'Learner':<15} | {'Elasticity':<10} | {'Std Error':<10} | {'t-stat':<10} | {'p-val':<10}")
    print("-" * 65)
    for name, model in results_dict.items():
        coeff = model.coef[0]
        stderr = model.se[0]
        tstat = model.t_stat[0]
        pval = model.pval[0]
        print(f"{name:<15} | {coeff:<10.4f} | {stderr:<10.4f} | {tstat:<10.4f} | {pval:<10.4f}")
