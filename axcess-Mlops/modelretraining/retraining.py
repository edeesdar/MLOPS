import boto3
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from sagemaker import session
from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.xgboost.estimator import XGBoost
import argparse
import csv
from io import StringIO


# --- Step 1: Initialize Resources -
#ml_s3_bucket = 'axcess-devst-sagemaker-bucket'



parser = argparse.ArgumentParser()
parser.add_argument("--kwargs", type=str, default=None)

args, _ = parser.parse_known_args()
parsed_kwargs = json.loads(args.kwargs)

model_package_group_name_input = parsed_kwargs.get('model_package_group_name_input')
model_version_input = parsed_kwargs.get('model_version_input')
ml_s3_bucket = parsed_kwargs.get('ml_s3_bucket')

prefix = 'taxi-duration'
base_s3_path = f"s3://{ml_s3_bucket}/data"
region = boto3.Session().region_name or 'us-east-1'
boto_session = boto3.Session(region_name=region)
sagemaker_session = session.Session(boto_session=boto_session)
s3_client = boto3.client('s3', region_name=region)
codepipeline_client = boto3.client('codepipeline', region_name=region)
role = "arn:aws:iam::345594592951:role/service-role/AmazonSageMaker-ExecutionRole-20250325T120134"
metrics_s3_key = f"{prefix}/metrics_csv/metrics.csv"
lambda_client = boto3.client('lambda')
Initial_Lambda_Switch = "mlops_evaluate_and_notify_lambda"



s3_config_path = f"{prefix}/data/config/monitoring_config.json"
config_obj = s3_client.get_object(Bucket=ml_s3_bucket, Key=s3_config_path)
config = json.loads(config_obj['Body'].read().decode('utf-8'))

# --- Step 3: Load Model Manifest from S3 ---
s3_manifest_path = f"{prefix}/data/config/model_manifest.json"
manifest_obj = s3_client.get_object(Bucket=ml_s3_bucket, Key=s3_manifest_path)
model_manifest = json.loads(manifest_obj['Body'].read().decode('utf-8'))


# --- Process Each Model ---
for model_info in model_manifest["models"]:
    model_name = f"{model_package_group_name_input}-{model_version_input}"
    
    # --- Check Retraining History ---


    # --- Evaluate Model Quality ---
    monitor = ModelQualityMonitor(
        role=role,
        sagemaker_session=sagemaker_session,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        volume_size_in_gb=20,
        max_runtime_in_seconds=1800
    )

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"{model_name}-quality-check-{current_time}"

    try:
        baseline_job = monitor.suggest_baseline(
            job_name=job_name,
            baseline_dataset=model_info["baseline_s3_path"],
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=f"s3://{ml_s3_bucket}/{prefix}/monitoring/results/{model_package_group_name_input}/{model_name}/",
            problem_type=model_info["model_type"],
            inference_attribute="prediction",
            ground_truth_attribute="ground_truth"
        )
        baseline_job.wait(logs=False)

        job = monitor.latest_baselining_job
        constraints = job.suggested_constraints()
        metrics = constraints.body_dict.get(f"{model_info['model_type'].lower()}_constraints", {})

        print("Constraint keys:", constraints.body_dict.keys())
        print("Full constraints:", json.dumps(constraints.body_dict, indent=2))

        print("Extracted metrics:",json.dumps(metrics, indent=2))

        #---------------------------------------------------------------------------------
        # Convert the metrics dictionary to CSV
        csv_buffer = StringIO()
        csv_writer = csv.writer(csv_buffer)
        
        # Write CSV header
        csv_writer.writerow(["metric", "threshold", "comparison_operator"])
        
        # Write each metric row
        for metric_name, details in metrics.items():
            csv_writer.writerow([metric_name, details["threshold"], details["comparison_operator"]])
        
        # Optionally, print the CSV data to check
        print("CSV content:")
        print(csv_buffer.getvalue())
        
        # Now upload the CSV to S3
        #s3 = boto3.client('s3')
        #bucket_name = 'your-s3-bucket'  # Replace with your bucket name
        #metrics_s3_key = 'path/to/metrics.csv'  # S3 key where the CSV will be stored
        
        response = s3_client.put_object(
            Bucket=ml_s3_bucket,
            Key=metrics_s3_key,
            Body=csv_buffer.getvalue()
        )
        
        print("CSV uploaded to s3://{}/{}".format(ml_s3_bucket, metrics_s3_key))

        #------------------------------------------------------------------------

        # Build event data
        event_payload = {
            "detail": {
                "model_name": model_name,
                "metrics": {}
            }
        }

        for metric_name in ["mae", "mse", "rmse"]:
            if metric_name in metrics:
                threshold_val = metrics[metric_name].get("threshold")
                event_payload["detail"]["metrics"][metric_name] = { "value": threshold_val }

        print("Prepared event payload:")
        print(json.dumps(event_payload, indent=2))

        # Trigger Lambda
        
        response = lambda_client.invoke(
            FunctionName=Initial_Lambda_Switch,  
            InvocationType='Event',  
            Payload=json.dumps(event_payload)
        )
        print("Lambda invoked:", response)




    except Exception as e:
        print(f"Monitoring failed for {model_name}: {str(e)}.")

       

    # --- Update Retraining History ---
