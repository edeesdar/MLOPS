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
role = "arn:aws:iam::317185619046:role/service-role/AmazonSageMaker-ExecutionRole-20240228T142935"
metrics_s3_key = f"{prefix}/metrics_csv/metrics.csv"



s3_config_path = f"{prefix}/data/config/monitoring_config.json"
config_obj = s3_client.get_object(Bucket=ml_s3_bucket, Key=s3_config_path)
config = json.loads(config_obj['Body'].read().decode('utf-8'))

# --- Step 3: Load Model Manifest from S3 ---
s3_manifest_path = f"{prefix}/data/config/model_manifest.json"
manifest_obj = s3_client.get_object(Bucket=ml_s3_bucket, Key=s3_manifest_path)
model_manifest = json.loads(manifest_obj['Body'].read().decode('utf-8'))

# --- Load Config and Manifest ---
#def load_s3_json(path):
#    bucket, key = path.replace("s3://", "").split("/", 1)
#    obj = s3_client.get_object(Bucket=bucket, Key=key)
#    return json.loads(obj["Body"].read().decode("utf-8"))

#model_manifest = load_s3_json(manifest_path)
#config = load_s3_json(config_path)

# --- Validate S3 Path ---
def validate_s3_path(path):
    if not path.startswith("s3://"):
        path = f"{base_s3_path}/{path.lstrip('/')}"
    bucket, key = path.replace("s3://", "").split("/", 1)
    s3_client.head_object(Bucket=bucket, Key=key)
    return path

# --- Validate Paths in Manifest ---
for model in model_manifest["models"]:
    model["baseline_s3_path"] = validate_s3_path(model["baseline_s3_path"])
    model["training_data_path"] = validate_s3_path(model["training_data_path"])

# --- Process Each Model ---
for model_info in model_manifest["models"]:
    model_name = f"{model_package_group_name_input}-{model_version_input}"
    
    # --- Check Retraining History ---
    history_key = f"{prefix}/monitoring/history/{model_package_group_name_input}/{model_name}_retraining_history.json"
    retrain = True

    try:
        obj = s3_client.get_object(Bucket=ml_s3_bucket, Key=history_key)
        history = json.loads(obj['Body'].read().decode('utf-8'))
        last_retraining = datetime.strptime(history["last_retraining"], "%Y-%m-%d %H:%M:%S")
        min_interval = timedelta(hours=config["retraining_policy"]["minimum_interval_hours"])
        retrain = (datetime.now() - last_retraining) >= min_interval
    except s3_client.exceptions.ClientError:
        pass

    if not retrain:
        print(f"Skipping retraining for {model_name}: minimum interval not met.")
        continue

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


        thresholds = config["quality_thresholds"][model_info["model_type"].lower()]
        failures = 0
        for metric, spec in thresholds.items():
            actual_value = metrics.get(metric, {}).get("threshold", 0)
            if spec["comparison"] == "LessThanThreshold" and actual_value > spec["threshold"]:
                failures += 1
            elif spec["comparison"] == "GreaterThanThreshold" and actual_value < spec["threshold"]:
                failures += 1

        if failures > 0:
            print(f"{model_name} failed quality check. Proceeding with retraining.")

            # --- Trigger CodePipeline ---
            try:
                training_pipeline_name = model_info["training_pipeline"]
                cp_response = codepipeline_client.start_pipeline_execution(name=training_pipeline_name)
                print(f"Triggered CodePipeline: {training_pipeline_name}, Execution ID: {cp_response['pipelineExecutionId']}")
            except Exception as e:
                print(f"Failed to trigger CodePipeline: {str(e)}")
        else:
            print(f"{model_name} passed quality check. Skipping retraining.")
            continue

    except Exception as e:
        print(f"Monitoring failed for {model_name}: {str(e)}.")

       

    # --- Update Retraining History ---
    history = {
        "model_package_group_name":model_package_group_name_input,
        "model_name": model_name,
        "last_retraining": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "retraining_count": 1
    }

    s3_client.put_object(
        Bucket=ml_s3_bucket,
        Key=history_key,
        Body=json.dumps(history).encode("utf-8")
    )
    print(f"Retraining history updated for {model_name}.")
