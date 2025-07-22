import argparse
import logging
import os
import glob
import traceback
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import numpy as np
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
def __read_data(path):
    try:
        all_files = glob.glob(os.path.join(path, "*.csv"))
        datasets = [pd.read_csv(f, header=None) for f in all_files]
        data = pd.concat(datasets, axis=0, ignore_index=True)
        y = data.iloc[:, 0]
        X = data.iloc[:, 1:]
        return X, y
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e
 
def calculate_rmse(y_true, y_pred):
    """Calculate RMSE in a version-agnostic way"""
    try:
        # Try new sklearn API first
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        # Fallback for older sklearn versions
        mse = mean_squared_error(y_true, y_pred)
        return math.sqrt(mse)
 
def _xgb_train(X_train, y_train, X_test, y_test, train_hp, is_master, model_dir):
    try:
        # Check XGBoost version to handle API differences
        xgb_version = xgb.__version__
        logger.info(f"Using XGBoost version: {xgb_version}")
        # Create the model
        xg_model = XGBRegressor(
            objective='reg:squarederror',
            max_depth=train_hp['max_depth'],
            learning_rate=train_hp['learning_rate'],
            n_estimators=train_hp['n_estimators'],
            gamma=train_hp['gamma'],
            min_child_weight=train_hp['min_child_weight'],
            subsample=train_hp['subsample'],
            colsample_bytree=train_hp['colsample_bytree'],
            reg_alpha=train_hp['reg_alpha'],
            reg_lambda=train_hp['reg_lambda']
        )
 
        # Try different fit methods based on XGBoost version
        fit_params = {
            'X': X_train,
            'y': y_train,
            'eval_set': [(X_test, y_test)],
            'verbose': True
        }
        try:
            # Try without early_stopping_rounds and eval_metric (newer versions)
            xg_model.fit(**fit_params)
        except TypeError as e:
            # If that fails, try with the old parameters
            logger.info(f"First fit attempt failed: {e}")
            logger.info("Trying with legacy parameters...")
            try:
                fit_params.update({
                    'eval_metric': 'rmse',
                    'early_stopping_rounds': 10
                })
                xg_model.fit(**fit_params)
            except TypeError as e2:
                # Final fallback - minimal fit
                logger.info(f"Second attempt failed: {e2}")
                logger.info("Using minimal fit parameters...")
                xg_model.fit(X_train, y_train)
 
        if is_master:
            model_location = os.path.join(model_dir, "xgboost-model")
            xg_model.get_booster().save_model(model_location)
            logger.info(f"Model saved at: {model_location}")
        return xg_model
 
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e
 
def _xgb_model_perf(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
 
    # Use version-agnostic RMSE calculation
    train_rmse = calculate_rmse(y_train, y_train_pred)
    test_rmse = calculate_rmse(y_test, y_test_pred)
 
    print("\nTraining Metrics:")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  MAE : {mean_absolute_error(y_train, y_train_pred):.2f}")
    print(f"  R2  : {r2_score(y_train, y_train_pred):.2f}")
 
    print("\nValidation Metrics:")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE : {mean_absolute_error(y_test, y_test_pred):.2f}")
    print(f"  R2  : {r2_score(y_test, y_test_pred):.2f}")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--min_child_weight', type=int, default=1)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    parser.add_argument('--reg_alpha', type=float, default=0)
    parser.add_argument('--reg_lambda', type=float, default=1)
 
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
 
    args = parser.parse_args()
    logger.info(f"Args: {args}")
 
    X_train, y_train = __read_data(args.train)
    X_test, y_test = __read_data(args.validation)
 
    train_hp = vars(args)
    model = _xgb_train(X_train, y_train, X_test, y_test, train_hp, is_master=True, model_dir=args.model_dir)
    _xgb_model_perf(model, X_train, X_test, y_train, y_test)