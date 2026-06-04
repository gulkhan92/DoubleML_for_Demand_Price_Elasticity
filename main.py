import os
import joblib
import pandas as pd
import config
from data_preparation import load_and_merge, filter_sparse_series
from feature_engineering import engineer_features
from model_training import run_double_ml
from evaluation import compare_with_ols, compare_dml_results

def main():
    print("Starting the Walmart Double ML Elasticity Pipeline...")
    
    # 1 & 2. Data Preparation & Feature Engineering (with Caching)
    if os.path.exists(config.FEATURES_PATH):
        print(f"\n--- Loading Cached Features from {config.FEATURES_PATH} ---")
        df = pd.read_pickle(config.FEATURES_PATH)
        print(f"Features loaded. Shape: {df.shape}")
    else:
        print("\n--- Step 1: Data Preparation ---")
        df = load_and_merge()
        df = filter_sparse_series(df)
        print(f"Data prepared. Final shape: {df.shape}")
        
        print("\n--- Step 2: Feature Engineering ---")
        df = engineer_features(df)
        
        os.makedirs(os.path.dirname(config.FEATURES_PATH), exist_ok=True)
        df.to_pickle(config.FEATURES_PATH)
        print(f"Features saved to {config.FEATURES_PATH}")
    
    # 3. Naive Baseline
    print("\n--- Step 3: Baseline Estimation ---")
    compare_with_ols(df)
    
    # 4. Double ML
    print("\n--- Step 4: Double ML Estimation ---")
    learners = ["lightgbm", "xgboost", "random_forest"]
    dml_results_dict = {}
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    for learner in learners:
        model_path = os.path.join(config.MODELS_DIR, f"{learner}_dml.joblib")
        if os.path.exists(model_path):
            print(f"Loading cached DML model for {learner}...")
            dml_results_dict[learner] = joblib.load(model_path)
        else:
            dml_results_dict[learner] = run_double_ml(df, learner_type=learner)
            joblib.dump(dml_results_dict[learner], model_path)
            print(f"DML model for {learner} saved to {model_path}")
    
    # 5. Final Report
    print("\n--- Step 5: Results Summary ---")
    compare_dml_results(dml_results_dict)
    print("\nPipeline execution complete.")

if __name__ == "__main__":
    main()
