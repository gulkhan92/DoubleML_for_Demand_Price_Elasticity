import pandas as pd
import numpy as np
import config

def load_and_merge():
    print("Loading data...")
    sales = pd.read_csv(config.SALES_PATH)
    prices = pd.read_csv(config.PRICES_PATH)
    calendar = pd.read_csv(config.CALENDAR_PATH)

    # Filter category
    sales = sales[sales['cat_id'] == config.TARGET_CATEGORY]

    # Melt sales to long format
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_long = sales.melt(id_vars=id_vars, var_name='d', value_name='sales')

    # Merge calendar
    df = sales_long.merge(calendar, on='d', how='left')
    
    # Fill event NaNs so they aren't dropped during aggregation
    df['event_name_1'] = df['event_name_1'].fillna('NoEvent')
    df['event_type_1'] = df['event_type_1'].fillna('NoEvent')

    # Aggregate to weekly level to match price data granularity
    # M5 sell_prices are weekly. We aggregate sales per store-item-week.
    # Removed 'wday' from groupby to achieve true weekly aggregation
    group_cols = ['item_id', 'store_id', 'wm_yr_wk', 'dept_id', 'state_id', 
                  'month', 'year', 'snap_CA', 'snap_TX', 'snap_WI', 
                  'event_name_1', 'event_type_1']
    df_weekly = df.groupby(group_cols).agg({
        'sales': 'sum',
        'date': 'first'
    }).reset_index()

    # Merge prices
    df_weekly = df_weekly.merge(prices, on=['item_id', 'store_id', 'wm_yr_wk'], how='inner')

    # Transformations
    df_weekly[config.Y_COL] = np.log1p(df_weekly['sales'])
    df_weekly[config.T_COL] = np.log(df_weekly['sell_price'])
    
    return df_weekly

def filter_sparse_series(df):
    counts = df.groupby(['item_id', 'store_id']).size()
    keep = counts[counts >= config.MIN_WEEKS_THRESHOLD].index
    return df.set_index(['item_id', 'store_id']).loc[keep].reset_index()
