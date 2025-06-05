import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import argparse
import logging
import traceback
import boto3
import awswrangler as wr


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

BASE_PATH = os.path.join("/", "opt", "ml")
PROCESSING_PATH = os.path.join(BASE_PATH, "processing")
#PROCESSING_PATH_INPUT = os.path.join(PROCESSING_PATH, "input")
PROCESSING_PATH_OUTPUT = os.path.join(PROCESSING_PATH, "output")

def transform_data(df):
    try:
        LOGGER.info("Original data shape: {}".format(df.shape))

        df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'], errors='coerce')
        df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'], errors='coerce')
        
        # Feature engineering
        df['pickup_hour'] = df['lpep_pickup_datetime'].dt.hour
        df['pickup_day'] = df['lpep_pickup_datetime'].dt.dayofweek
        df['pickup_month'] = df['lpep_pickup_datetime'].dt.month
        df['dropoff_hour'] = df['lpep_dropoff_datetime'].dt.hour
        df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds()
        
        # Drop original datetime columns
        df.drop(['lpep_pickup_datetime', 'lpep_dropoff_datetime'], axis=1, inplace=True)
        
        # Handle categorical variables
        categorical_cols = ['VendorID', 'store_and_fwd_flag', 'RatecodeID', 
                           'PULocationID', 'DOLocationID', 'payment_type']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        # Handle missing values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        # Ensure all columns are numeric
        df = df.select_dtypes(include=['number'])
        
        return df
    
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))
        raise e

def load_data(df, file_path, file_name, header=True):
    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        path = os.path.join(file_path, file_name + ".csv")
        LOGGER.info("Saving file to {}".format(path))
        df.to_csv(path, index=False, header=header)
        
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Read input data
    #input_files = [f for f in os.listdir(PROCESSING_PATH_INPUT) 
    #             if f.endswith('.csv')]
    #dfs = [pd.read_csv(os.path.join(PROCESSING_PATH_INPUT, f), 
    #      parse_dates=['lpep_pickup_datetime', 'lpep_dropoff_datetime'])
    #      for f in input_files]
    session = boto3.Session(region_name="us-east-1")  # e.g., "us-east-1"

    df = wr.athena.read_sql_query("SELECT * FROM mlops_pipeline_table_2018 LIMIT 10000",
    database="mlops_pipeline_database",
    boto3_session=session,
    s3_output="s3://mlops-query-result-bucket/"
    )
    
    #df = pd.concat(dfs)
    df = transform_data(df)
    
    # Split data
    target = 'trip_duration'
    X = df.drop(target, axis=1)
    y = df[target]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Save data
    train_data = pd.concat([y_train, X_train], axis=1)
    validation_data = pd.concat([y_test, X_test], axis=1)
    
    load_data(train_data, os.path.join(PROCESSING_PATH_OUTPUT, "train"), "train", header=False)
    load_data(validation_data, os.path.join(PROCESSING_PATH_OUTPUT, "validation"), "validation", header=False)
    load_data(X_test, os.path.join(PROCESSING_PATH_OUTPUT, "inference"), "data", header=True)
