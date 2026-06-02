import os

# Data Paths
DATA_DIR = "data"
SALES_PATH = os.path.join(DATA_DIR, "sales_train_evaluation.csv")
PRICES_PATH = os.path.join(DATA_DIR, "sell_prices.csv")
CALENDAR_PATH = os.path.join(DATA_DIR, "calendar.csv")

# Project Scope
TARGET_CATEGORY = "FOODS"
MIN_WEEKS_THRESHOLD = 10

# Column Names
Y_COL = "log_quantity"
T_COL = "log_price"
DATE_COL = "date"

# Features
# X: Effect modifiers (used for heterogeneity)
X_COLS = ['dept_id', 'store_id', 'state_id']

# W: Controls (Confounders)
W_COLS = [
    'month', 'year', 'snap_CA', 'snap_TX', 'snap_WI',
    'event_name_1', 'event_type_1', 'lag_sales_1', 'lag_sales_2'
]

# ML Params
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "importance_type": "gain",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1
}
