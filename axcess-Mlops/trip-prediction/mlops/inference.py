import json
import pickle as pkl
import xgboost as xgb
import tempfile
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from io import StringIO
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


print("\n# Part 8: Batch Transform Job")
print("=" * 40)

# Import necessary libraries
import boto3
import sagemaker
from sagemaker.transformer import Transformer
from sagemaker import get_execution_role
import pandas as pd
import numpy as np
from datetime import datetime
import os
import tarfile
import io
import sagemaker.session
import time

# Import image_uris for retrieving the correct inference image
from sagemaker import image_uris

# Initialize SageMaker session and client
region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
default_bucket = "axcess-devst-sagemaker-bucket"

boto_sess = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=boto_sess)
role = get_execution_role()
sagemaker_client = boto_sess.client("sagemaker")
s3_client = boto_sess.client('s3')

# Get the proper inference image for XGBoost 1.7-1
inference_image = image_uris.retrieve(
    framework='xgboost',
    region=region,
    version='1.7-1',
    image_scope='inference'
)

# Define S3 locations
s3_bucket = "axcess-devst-sagemaker-bucket"
s3_prefix = "taxi-duration"
model_s3_path = f"s3://{s3_bucket}/{s3_prefix}/models/model.tar.gz"
batch_input_path = f"s3://{s3_bucket}/{s3_prefix}/batch_input/"
batch_output_path = f"s3://{s3_bucket}/{s3_prefix}/batch_output/"  # Updated to match your output path
model_name = "trip-prediction-7"

# Step 1: Prepare batch input data
print("\nPreparing batch input data...")
test_data_path = f"s3://{s3_bucket}/{s3_prefix}/test/data.csv"
test_data = pd.read_csv(test_data_path, header=None, storage_options={})
print(f"Test data shape: {test_data.shape}")

inference_data = test_data.iloc[:100, 1:]  # Take first 100 rows, exclude target
print(f"Inference data shape: {inference_data.shape}")

# Upload inference data to S3
s3_path = f"{batch_input_path}inference_data.csv"
inference_data.to_csv(s3_path, index=False, header=False, storage_options={})
print(f"Uploaded inference data to {s3_path}")
output_file_key = f"{s3_prefix}/batch_output/inference_data.csv.out"
#output_file_key = f"{s3_prefix}/test/data/output/inference_data.csv.out"

# Step 2: Create and run the batch transform job


# Step 3: Download and process the batch transform results
#def check_for_output_file():
#    """Check if the output file exists with retries"""
#    #output_file_key = f"{s3_prefix}/test/data/output/inference_data.csv.out"
#    output_file_key = "s3://axcess-devst-sagemaker-bucket/taxi-duration/batch_output/inference_data.csv.out"
#    max_attempts = 10
#    wait_time = 30
#    
#    for attempt in range(max_attempts):
#        try:
#            s3_client.head_object(Bucket=s3_bucket, Key=output_file_key)
#            print(f"Found output file at s3://{s3_bucket}/{output_file_key}")
#            return output_file_key
#        except s3_client.exceptions.ClientError as e:
#            if e.response['Error']['Code'] == '404':
#                print(f"Attempt {attempt + 1}: Output not yet available, waiting {wait_time} seconds...")
#                time.sleep(wait_time)
#            else:
#                print(f"S3 Error: {str(e)}")
#                raise
#    return None

try:
    # Check for the output file with retries
    #output_file_key = check_for_output_file()
    
    if output_file_key:
        # Download and read the predictions
        print('s3_bucket:',s3_bucket)
        print('output_file_key:',output_file_key)
        s3_object = s3_client.get_object(Bucket=s3_bucket, Key=output_file_key)
        predictions = pd.read_csv(io.BytesIO(s3_object['Body'].read()), header=None)
        print(f"Predictions shape: {predictions.shape}")
        
        print("\nSample predictions (seconds):")
        print(predictions.head())
        
        # Convert to minutes
        predictions_minutes = predictions.copy()
        predictions_minutes[0] = predictions_minutes[0] / 60
        print("\nSample predictions (minutes):")
        print(predictions_minutes.head())
        
        # Save predictions in minutes to S3
        output_csv_buffer = io.StringIO()
        predictions_minutes.to_csv(output_csv_buffer, index=False, header=False)
        predictions_minutes_key = f"{s3_prefix}/predictions/predictions_minutes.csv"
        s3_client.put_object(Bucket=s3_bucket, Key=predictions_minutes_key, Body=output_csv_buffer.getvalue())
        print(f"Saved predictions in minutes to s3://{s3_bucket}/{predictions_minutes_key}")
        
        # Combine with original features
        original_data = inference_data.iloc[:predictions.shape[0], :]
        combined_results = pd.concat([predictions, original_data.reset_index(drop=True)], axis=1)
        combined_results.columns = ['predicted_duration_seconds'] + [f'feature_{i}' for i in range(original_data.shape[1])]
        combined_results['predicted_duration_minutes'] = combined_results['predicted_duration_seconds'] / 60
        
        # Save combined results to S3
        combined_csv_buffer = io.StringIO()
        combined_results.to_csv(combined_csv_buffer, index=False)
        combined_results_key = f"{s3_prefix}/predictions/predictions_with_features.csv"
        s3_client.put_object(Bucket=s3_bucket, Key=combined_results_key, Body=combined_csv_buffer.getvalue())
        print(f"Saved combined results to s3://{s3_bucket}/{combined_results_key}")
        
        # Step 4: Calculate metrics
        #------------------------------------------------------------------------

        categorical_cols = test_data.select_dtypes(include=['object', 'category']).columns

        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            test_data[col] = le.fit_transform(test_data[col])
            label_encoders[col] = le  # Save for later use if needed

        #---------------------------------------------------------------------------------
        actual_values = test_data.iloc[:predictions.shape[0], 0]
        
        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values, predictions)
        r2 = r2_score(actual_values, predictions)
        
        print("\nBatch Transform Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
            'Value': [mse, rmse, mae, r2]
        })

        metrics_csv_buffer = io.StringIO()
        metrics_df.to_csv(metrics_csv_buffer, index=False)
        metrics_key = f"{s3_prefix}/metrics/batch_transform_metrics.csv"
        s3_client.put_object(Bucket=s3_bucket, Key=metrics_key, Body=metrics_csv_buffer.getvalue())
        print(f"Uploaded metrics to s3://{s3_bucket}/{metrics_key}")
    else:
        print("Error: Output file not found after multiple attempts")
except Exception as e:
    print(f"Error processing batch transform results: {e}")

print("\n# Summary")
print("=" * 40)
print("Batch transform job complete!")
print(f"Model used: {model_name}")
#print(f"Batch transform job: {transform_job_name}")
print(f"Input data: {batch_input_path}inference_data.csv")
print(f"Output data: {batch_output_path}")
