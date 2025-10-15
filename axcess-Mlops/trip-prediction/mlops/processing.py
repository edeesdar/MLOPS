import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import logging
import traceback
import boto3
from datetime import datetime
 
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
 
BASE_PATH = os.path.join("/", "opt", "ml")
PROCESSING_PATH = os.path.join(BASE_PATH, "processing")
PROCESSING_PATH_INPUT = os.path.join(PROCESSING_PATH, "input")
PROCESSING_PATH_OUTPUT = os.path.join(PROCESSING_PATH, "output")
 
s3_bucket = "mlops-workshop-edees-nasrullah-dar"
input_s3_key = "taxi-duration/original_raw_data/data.csv"
prefix = "taxi-duration"
 
def transform_data(df):
    try:
        LOGGER.info("Original data shape: {}".format(df.shape))
        df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'], errors='coerce')
        df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'], errors='coerce')
 
        # Filter out rows with invalid datetimes
        df.dropna(subset=['lpep_pickup_datetime', 'lpep_dropoff_datetime'], inplace=True)
 
        # Feature engineering
        df['pickup_hour'] = df['lpep_pickup_datetime'].dt.hour
        df['pickup_day'] = df['lpep_pickup_datetime'].dt.dayofweek
        df['pickup_month'] = df['lpep_pickup_datetime'].dt.month
        df['dropoff_hour'] = df['lpep_dropoff_datetime'].dt.hour
        df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds()
 
        # Remove rows with invalid duration
        df = df[(df['trip_duration'] > 0) & (df['trip_duration'] < 7200)]  # max 2 hours
 
        df.drop(['lpep_pickup_datetime', 'lpep_dropoff_datetime'], axis=1, inplace=True)
 
        categorical_cols = ['VendorID', 'store_and_fwd_flag', 'RatecodeID', 
                            'PULocationID', 'DOLocationID', 'payment_type']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
 
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
 
        df = df.select_dtypes(include=['number'])
 
        return df
    except Exception as e:
        LOGGER.error(traceback.format_exc())
        raise e
 
def load_data(df, file_path, file_name, header=True):
    try:
        os.makedirs(file_path, exist_ok=True)
        path = os.path.join(file_path, file_name + ".csv")
        LOGGER.info(f"Saving file to {path}")
        df.to_csv(path, index=False, header=header)
    except Exception as e:
        LOGGER.error(traceback.format_exc())
        raise e
 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
 
    s3_uri = f"s3://{s3_bucket}/{input_s3_key}"
    try:
        df = pd.read_csv(s3_uri)
        LOGGER.info(f"Loaded from S3: {s3_uri} with shape {df.shape}")
    except Exception as e:
        LOGGER.error(f"Failed to read from S3: {e}")
        raise
 
    df = transform_data(df)
 
    # Save original trip_duration sample
    local_dir = "/tmp/processed"
    os.makedirs(local_dir, exist_ok=True)
    trip_duration_df = df[['trip_duration']].head(100)
    trip_duration_file = os.path.join(local_dir, "original_trip_duration.csv")
    trip_duration_df.to_csv(trip_duration_file, index=False, header=True)
 
    s3 = boto3.resource('s3')
    s3.Object(s3_bucket, f"{prefix}/output/original_trip_duration.csv").upload_file(trip_duration_file)
    print(f"Uploaded to: s3://{s3_bucket}/{prefix}/output/original_trip_duration.csv")
 
    # Split
    from sklearn.model_selection import train_test_split
    X = df.drop("trip_duration", axis=1)
    y = df["trip_duration"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
    train_data = pd.concat([y_train, X_train], axis=1)
    validation_data = pd.concat([y_test, X_test], axis=1)
 
    load_data(train_data, os.path.join(PROCESSING_PATH_OUTPUT, "train"), "train", header=False)
    load_data(validation_data, os.path.join(PROCESSING_PATH_OUTPUT, "validation"), "validation", header=False)
    load_data(X_test, os.path.join(PROCESSING_PATH_OUTPUT, "inference"), "data", header=True)

