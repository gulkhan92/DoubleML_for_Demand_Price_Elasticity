import os
import pandas as pd
import config
from model_training import run_double_ml
from feature_engineering import engineer_features
from data_preparation import load_and_merge, filter_sparse_series

def run_heterogeneity_analysis():
    print(f"--- Starting Heterogeneity Analysis by {config.HETERO_SEGMENT_COL} ---")
    
    # Load data
    if os.path.exists(config.FEATURES_PATH):
        df = pd.read_pickle(config.FEATURES_PATH)
    else:
        df = load_and_merge()
        df = filter_sparse_series(df)
        df = engineer_features(df)

    # Mapping back category codes might be needed for readability
    # But for this practice, we iterate unique codes found in the segment column
    segments = df[config.HETERO_SEGMENT_COL].unique()
    results = []

    for segment in segments:
        print(f"\nProcessing Segment: {segment}")
        segment_df = df[df[config.HETERO_SEGMENT_COL] == segment].copy()
        
        if len(segment_df) < 5000: # Minimum sample size for a reliable DML estimate
            print(f"Skipping segment {segment}: Insufficient data.")
            continue
            
        try:
            dml_model = run_double_ml(segment_df, learner_type="lightgbm")
            results.append({
                'segment': segment,
                'elasticity': dml_model.coef[0],
                'se': dml_model.se[0],
                'pval': dml_model.pval[0]
            })
        except Exception as e:
            print(f"Failed to estimate for segment {segment}: {e}")

    results_df = pd.DataFrame(results)
    print("\n--- Heterogeneity Summary ---")
    print(results_df.sort_values('elasticity'))
    
    print(f"\nSpread: Min {results_df['elasticity'].min():.4f} to Max {results_df['elasticity'].max():.4f}")

if __name__ == "__main__":
    run_heterogeneity_analysis()