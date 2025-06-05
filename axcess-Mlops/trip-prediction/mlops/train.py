import argparse
import logging
import os
import pickle as pkl
import glob
import traceback
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
#from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import boto3
import numpy as np



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def __read_data(files_path):
    try:
        logger.info("Reading dataset from source...")
        all_files = glob.glob(os.path.join(files_path, "*.csv"))
        
        datasets = []
        for filename in all_files:
            data = pd.read_csv(filename, header=None)
            datasets.append(data)
            
        data = pd.concat(datasets, axis=0, ignore_index=True)
        y = data.iloc[:, 0]
        X = data.iloc[:, 1:]
        
        return X, y
        
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))
        raise e

def _xgb_train(X_train, y_train, X_test, y_test, train_hp, is_master, model_dir):
    try:
        logger.info("Training the Model...")
        
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
        
        xg_model.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    #early_stopping_rounds=10,
                    verbose=True)

        if is_master:
            model_location = os.path.join(model_dir, 'xgboost-model')
            #pkl.dump(xg_model, open(model_location, 'wb'))
            #xg_model.save_model(model_location)
            booster = xg_model.get_booster()
            booster.save_model(model_location)
            logger.info("Stored trained model at {}".format(model_location))
            #xg_model = xgb.Booster()
            #xg_model.load_model(model_location)
            
        return xg_model
        
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))
        raise e

def _xgb_model_perf(xg_model, X_train, X_test, y_train, y_test):
    try:
        y_train_pred = xg_model.predict(X_train)
        y_test_pred = xg_model.predict(X_test)
        
        print("\nModel Performance:")
        print("="*40)
        print("Training Metrics:")
        print(f"- RMSE: {mean_squared_error(y_train, y_train_pred):.2f}")
        print(f"- MAE: {mean_absolute_error(y_train, y_train_pred):.2f}")
        print(f"- R2: {r2_score(y_train, y_train_pred):.2f}")
        
        print("\nTest Metrics:")
        print(f"- RMSE: {mean_squared_error(y_test, y_test_pred):.2f}")
        print(f"- MAE: {mean_absolute_error(y_test, y_test_pred):.2f}")
        print(f"- R2: {r2_score(y_test, y_test_pred):.2f}")
        
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--min_child_weight', type=int, default=1)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    parser.add_argument('--reg_alpha', type=float, default=0)
    parser.add_argument('--reg_lambda', type=float, default=1)
    
    # SageMaker specific arguments
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    args = parser.parse_args()
    logger.info("Arguments: {}".format(args))
    
    # Read data
    X_train, y_train = __read_data(args.train)
    X_test, y_test = __read_data(args.validation)
    
    # Train model
    train_hp = {
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'reg_alpha': args.reg_alpha,
        'reg_lambda': args.reg_lambda
    }
    
    xgb_model = _xgb_train(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_hp=train_hp,
        is_master=True,
        model_dir=args.model_dir
    )
    
    # Evaluate model
    _xgb_model_perf(xgb_model, X_train, X_test, y_train, y_test)
