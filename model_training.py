import doubleml as dml
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor, XGBRFRegressor
import config

def run_double_ml(df, learner_type="lightgbm"):
    print(f"Initializing Double ML with {learner_type}...")
    
    # Define the data structure for DoubleML
    # X = effect modifiers, W = controls
    obj_dml_data = dml.DoubleMLData(
        df,
        y_col=config.Y_COL,
        d_cols=config.T_COL,
        x_cols=config.X_COLS + config.W_COLS
    )

    # Nuisance Models
    if learner_type == "lightgbm":
        ml_l = LGBMRegressor(**config.LGBM_PARAMS)
        ml_m = LGBMRegressor(**config.LGBM_PARAMS)
    elif learner_type == "xgboost":
        ml_l = XGBRegressor(**config.XGB_PARAMS)
        ml_m = XGBRegressor(**config.XGB_PARAMS)
    elif learner_type == "random_forest":
        ml_l = XGBRFRegressor(**config.RF_PARAMS)
        ml_m = XGBRFRegressor(**config.RF_PARAMS)
    else:
        raise ValueError(f"Unsupported learner type: {learner_type}")

    # DML2 (Partialling Out)
    dml_plr = dml.DoubleMLPLR(obj_dml_data, ml_l, ml_m, n_folds=5)
    
    print("Fitting DML model (this may take a while)...")
    dml_plr.fit()
    
    return dml_plr
