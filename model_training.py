import doubleml as dml
from lightgbm import LGBMRegressor
import config

def run_double_ml(df):
    print("Initializing Double ML...")
    
    # Define the data structure for DoubleML
    # X = effect modifiers, W = controls
    obj_dml_data = dml.DoubleMLData(
        df,
        y_col=config.Y_COL,
        d_cols=config.T_COL,
        x_cols=config.X_COLS + config.W_COLS
    )

    # Nuisance Models
    ml_l = LGBMRegressor(**config.LGBM_PARAMS) # E[Y | X, W]
    ml_m = LGBMRegressor(**config.LGBM_PARAMS) # E[T | X, W]

    # DML2 (Partialling Out)
    dml_plr = dml.DoubleMLPLR(obj_dml_data, ml_l, ml_m, n_folds=5)
    
    print("Fitting DML model (this may take a while)...")
    dml_plr.fit()
    
    return dml_plr
