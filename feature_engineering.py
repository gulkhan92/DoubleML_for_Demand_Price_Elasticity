import pandas as pd
import config

def engineer_features(df):
    print("Engineering features...")
    df = df.sort_values(['item_id', 'store_id', 'wm_yr_wk'])

    # Lagged sales (crucial for demand patterns)
    df['lag_sales_1'] = df.groupby(['item_id', 'store_id'])[config.Y_COL].shift(1)
    df['lag_sales_2'] = df.groupby(['item_id', 'store_id'])[config.Y_COL].shift(2)
    
    # Fill NaNs from lags
    df = df.dropna(subset=['lag_sales_1', 'lag_sales_2'])

    # Handle Categorical variables for ML models
    cat_cols = config.X_COLS + ['event_name_1', 'event_type_1']
    for col in cat_cols:
        df[col] = df[col].fillna('unknown').astype('category').cat.codes
        
    return df
