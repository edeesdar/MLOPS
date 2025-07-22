 
import os
import boto3
import json
import time
import pandas as pd
from sagemaker import session
from sagemaker.model_monitor import DefaultModelMonitor, CronExpressionGenerator
from sagemaker.model_monitor import BatchTransformInput
from sagemaker.model_monitor.dataset_format import MonitoringDatasetFormat
from time import gmtime, strftime

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from io import StringIO
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sagemaker
#from sagemaker.transformer import Transformer
from sagemaker import get_execution_role
import numpy as np
from datetime import datetime
import tarfile
import io
import sagemaker.session
import time

from sagemaker import image_uris

import argparse
import csv


 
# Initialize configuration
region = boto3.Session().region_name
boto_sess = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=boto_sess)
role = get_execution_role()
sagemaker_client = boto_sess.client("sagemaker")
s3_client = boto_sess.client('s3')

parser = argparse.ArgumentParser()
parser.add_argument("--kwargs", type=str, default=None)

args, _ = parser.parse_known_args()
parsed_kwargs = json.loads(args.kwargs)

model_package_group_name_input = parsed_kwargs.get('model_package_group_name_input')
model_version_input = parsed_kwargs.get('model_version_input')
ml_s3_bucket = parsed_kwargs.get('ml_s3_bucket')
 
# Handle role determination safely
try:
    from sagemaker import get_execution_role
    role = get_execution_role()
except Exception as e:
    print(f"Couldn't get execution role automatically: {str(e)}")
    # Using your provided role ARN
    role = "arn:aws:iam::345594592951:role/AmazonSageMakerServiceCatalogProductsUseRoleMultiModelTB"
   
print(f"Using RoleArn: {role}")
 
# Your specific bucket and paths
#bucket = "axcess-devst-sagemaker-bucket"
#s3_prefix
#s3_bucket
prefix = "taxi-duration"

#------------------------Trip prediction in min and sec-------------------


model_s3_path = f"s3://{ml_s3_bucket}/{prefix}/models/model.tar.gz"
batch_input_path = f"s3://{ml_s3_bucket}/{prefix}/batch_input/"
batch_output_path = f"s3://{ml_s3_bucket}/{prefix}/batch_output/"  # Updated to match your output path
#model_name = "trip-prediction-3"

# Step 1: Prepare batch input data
print("\nPreparing batch input data...")
test_data_path = f"s3://{ml_s3_bucket}/{prefix}/test/data.csv"
test_data = pd.read_csv(test_data_path, header=None, storage_options={})
print(f"Test data shape: {test_data.shape}")

inference_data = test_data.iloc[:100, :]  # Take first 100 rows, exclude target
print(f"Inference data shape: {inference_data.shape}")

# Upload inference data to S3
#s3_path = f"{batch_input_path}inference_data.csv"
#inference_data.to_csv(s3_path, index=False, header=False, storage_options={})
#print(f"Uploaded inference data to {s3_path}")
output_file_key = f"{prefix}/batch_output/inference_data.csv.out"

time.sleep(120)
print('sleep time completed')

try:
    # Check for the output file with retries
    #output_file_key = check_for_output_file()
    
    if output_file_key:
        # Download and read the predictions
        s3_object = s3_client.get_object(Bucket=ml_s3_bucket, Key=output_file_key)
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
        predictions_minutes_key = f"{prefix}/predictions/predictions_minutes.csv"
        s3_client.put_object(Bucket=ml_s3_bucket, Key=predictions_minutes_key, Body=output_csv_buffer.getvalue())
        print(f"Saved predictions in minutes to s3://{ml_s3_bucket}/{predictions_minutes_key}")
        
        # Combine with original features
        original_data = inference_data.iloc[:predictions.shape[0], :]
        combined_results = pd.concat([predictions, original_data.reset_index(drop=True)], axis=1)
        combined_results.columns = ['predicted_duration_seconds'] + [f'feature_{i}' for i in range(original_data.shape[1])]
        combined_results['predicted_duration_minutes'] = combined_results['predicted_duration_seconds'] / 60
        
        # Save combined results to S3
        combined_csv_buffer = io.StringIO()
        combined_results.to_csv(combined_csv_buffer, index=False)
        combined_results_key = f"{prefix}/predictions/predictions_with_features.csv"
        s3_client.put_object(Bucket=ml_s3_bucket, Key=combined_results_key, Body=combined_csv_buffer.getvalue())
        print(f"Saved combined results to s3://{ml_s3_bucket}/{combined_results_key}")
        
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
        metrics_key = f"{prefix}/metrics/batch_transform_metrics.csv"
        s3_client.put_object(Bucket=ml_s3_bucket, Key=metrics_key, Body=metrics_csv_buffer.getvalue())
        print(f"Uploaded metrics to s3://{ml_s3_bucket}/{metrics_key}")
    else:
        print("Error: Output file not found after multiple attempts")
except Exception as e:
    print(f"Error processing batch transform results: {e}")

print("\n# Summary")
print("=" * 40)
print("Batch transform job complete!")
#print(f"Model used: {model_name}")
#print(f"Batch transform job: {transform_job_name}")
print(f"Input data: {batch_input_path}inference_data.csv")
print(f"Output data: {batch_output_path}")


#------------------------Complete code------------------------------------

 
# Paths for data capture and monitoring reports
data_capture_prefix = f"{prefix}/datacapture"
s3_capture_upload_path = f"s3://{ml_s3_bucket}/{data_capture_prefix}"
reports_prefix = f"{prefix}/reports"
s3_report_path = f"s3://{ml_s3_bucket}/{reports_prefix}"
 
print(f"Capture path: {s3_capture_upload_path}")
print(f"Report path: {s3_report_path}")
 
# Initialize SageMaker session
sagemaker_session = session.Session()
 
# 1) Prepare Baseline Data
baseline_prefix = f"{prefix}/baselining"
baseline_data_prefix = f"{baseline_prefix}/data"
baseline_results_prefix = f"{baseline_prefix}/results"
 
baseline_data_uri = f"s3://{ml_s3_bucket}/{baseline_data_prefix}"
baseline_results_uri = f"s3://{ml_s3_bucket}/{baseline_results_prefix}"
print(f"Baseline data uri: {baseline_data_uri}")
print(f"Baseline results uri: {baseline_results_uri}")
 
# 2) Create Baseline Job using your existing batch output
my_default_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sagemaker_session
)
 
# Use your existing batch output as baseline
baseline_dataset = f"s3://{ml_s3_bucket}/{prefix}/batch_output/inference_data.csv.out"
 
print("Creating baseline from existing batch output...")
my_default_monitor.suggest_baseline(
    baseline_dataset=baseline_dataset,
    dataset_format=MonitoringDatasetFormat.csv(header=False),  # No header in your output
    output_s3_uri=baseline_results_uri,
    wait=True,
)
 
# 3) Create Monitoring Schedule
statistics_path = f"{baseline_results_uri}/statistics.json"
constraints_path = f"{baseline_results_uri}/constraints.json"
 
mon_schedule_name = "green-taxi-batch-monitor-schedule-" + strftime(
    "%Y-%m-%d-%H-%M-%S", gmtime()
)
 
# Define the correct input path for monitoring
monitoring_input_path = f"s3://{ml_s3_bucket}/{prefix}/model_monitor/input/monitoring_dataset.csv"
 
print("Creating monitoring schedule...")
my_default_monitor.create_monitoring_schedule(
    monitor_schedule_name=mon_schedule_name,
    batch_transform_input=BatchTransformInput(
        data_captured_destination_s3_uri=monitoring_input_path,
        destination="/opt/ml/processing/input",
        dataset_format=MonitoringDatasetFormat.csv(header=False),
    ),
    output_s3_uri=s3_report_path,
    statistics=statistics_path,
    constraints=constraints_path,
    schedule_cron_expression=CronExpressionGenerator.hourly(),
    enable_cloudwatch_metrics=True,
)
 
# 4) Verify the schedule
print("\nMonitoring setup complete!")
print(f"Schedule Name: {mon_schedule_name}")
print(f"Results will be saved to: {s3_report_path}")
print(f"To check status later, run: my_default_monitor.describe_schedule()")
 
