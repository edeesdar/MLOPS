 
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
 
# Initialize configuration
region = boto3.Session().region_name
 
# Handle role determination safely
try:
    from sagemaker import get_execution_role
    role = get_execution_role()
except Exception as e:
    print(f"Couldn't get execution role automatically: {str(e)}")
    # Using your provided role ARN
    role = "arn:aws:iam::317185619046:role/AmazonSageMakerServiceCatalogProductsUseRoleMultiModelTB"
   
print(f"Using RoleArn: {role}")
 
# Your specific bucket and paths
bucket = "axcess-devst-sagemaker-bucket"
prefix = "taxi-duration"
 
# Paths for data capture and monitoring reports
data_capture_prefix = f"{prefix}/datacapture"
s3_capture_upload_path = f"s3://{bucket}/{data_capture_prefix}"
reports_prefix = f"{prefix}/reports"
s3_report_path = f"s3://{bucket}/{reports_prefix}"
 
print(f"Capture path: {s3_capture_upload_path}")
print(f"Report path: {s3_report_path}")
 
# Initialize SageMaker session
sagemaker_session = session.Session()
 
# 1) Prepare Baseline Data
baseline_prefix = f"{prefix}/baselining"
baseline_data_prefix = f"{baseline_prefix}/data"
baseline_results_prefix = f"{baseline_prefix}/results"
 
baseline_data_uri = f"s3://{bucket}/{baseline_data_prefix}"
baseline_results_uri = f"s3://{bucket}/{baseline_results_prefix}"
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
baseline_dataset = "s3://axcess-devst-sagemaker-bucket/taxi-duration/batch_output/inference_data.csv.out"
 
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
monitoring_input_path = "s3://axcess-devst-sagemaker-bucket/taxi-duration/model_monitor/input/monitoring_dataset.csv"
 
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
 
