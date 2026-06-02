from data_preparation import load_and_merge, filter_sparse_series
from feature_engineering import engineer_features
from model_training import run_double_ml
from evaluation import compare_with_ols, print_dml_summary

def main():
    print("Starting the Walmart Double ML Elasticity Pipeline...")
    
    # 1. Prepare
    print("\n--- Step 1: Data Preparation ---")
    df = load_and_merge()
    df = filter_sparse_series(df)
    print(f"Data prepared. Final shape: {df.shape}")
    
    # 2. Features
    print("\n--- Step 2: Feature Engineering ---")
    df = engineer_features(df)
    
    # 3. Naive Baseline
    print("\n--- Step 3: Baseline Estimation ---")
    compare_with_ols(df)
    
    # 4. Double ML
    print("\n--- Step 4: Double ML Estimation ---")
    dml_results = run_double_ml(df)
    
    # 5. Final Report
    print("\n--- Step 5: Results Summary ---")
    print_dml_summary(dml_results)
    print("\nPipeline execution complete.")

if __name__ == "__main__":
    main()
