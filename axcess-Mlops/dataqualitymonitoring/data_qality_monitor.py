import boto3
import pandas as pd
from sagemaker import session
from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
import os
from datetime import datetime
import argparse
import csv
from io import StringIO
import json

#s3_bucket = "axcess-devst-sagemaker-bucket"
s3_prefix = "taxi-duration"
#prefix

parser = argparse.ArgumentParser()
parser.add_argument("--kwargs", type=str, default=None)

args, _ = parser.parse_known_args()
parsed_kwargs = json.loads(args.kwargs)

model_package_group_name_input = parsed_kwargs.get('model_package_group_name_input')
model_version_input = parsed_kwargs.get('model_version_input')
ml_s3_bucket = parsed_kwargs.get('ml_s3_bucket')
 
def get_sagemaker_session(region_name):
    boto_session = boto3.Session(region_name=region_name)
    return session.Session(boto_session=boto_session)
 
def initialize_resources():
    try:
        region = boto3.Session().region_name or 'us-east-1'
        sagemaker_session = get_sagemaker_session(region)
        role = "arn:aws:iam::345594592951:role/AmazonSageMakerServiceCatalogProductsUseRoleMultiModelTB"
        print(f"Using IAM role: {role}")
        return sagemaker_session, region, None, role
    except Exception as e:
        print(f"Error initializing AWS resources: {str(e)}")
        raise
 

 
def prepare_baseline_dataset(ml_s3_bucket, s3_prefix):
    try:
        os.makedirs("processed", exist_ok=True)
        # Load ground truth data
        gt_path = f"s3://{ml_s3_bucket}/{s3_prefix}/model_monitor/input/monitoring_dataset.csv"
        column_name = "ground_truth"
        #gt_data, gt_na = load_dataset_with_header(gt_path, "ground_truth")
        """Load a specific column from a CSV with headers"""
        try:
            print(f"Loading column '{column_name}' from: {gt_path}")
            gt_data = pd.read_csv(
                gt_path,
                usecols=[column_name],
                engine='python'
            )
            # Convert to numeric, coercing errors to NaN
            gt_data[column_name] = pd.to_numeric(gt_data[column_name], errors='coerce')
            gt_na = gt_data[column_name].isna().sum()
            gt_clean = gt_data.dropna(subset=['ground_truth'])
            #return gt_df, na_count
        except Exception as e:
            print(f"Error loading column '{column_name}': {str(e)}")
            raise
        
        print(f"Loaded {len(gt_clean)} ground truth records, {gt_na} invalid values")

        # Load prediction data
        pred_path = f"s3://{ml_s3_bucket}/{s3_prefix}/batch_output/inference_data.csv.out"
        column_name = "prediction"
        #pred_data, pred_na = load_dataset_with_header(pred_path, "prediction")

        """Load a specific column from a CSV with headers"""
        try:
            print(f"Loading data from: {pred_path}")
            pred_data = pd.read_csv(
                    pred_path,
                    header=None,               # No header in the file
                    names=["prediction"]       # Assign column name
            )

           #------------------------------------------------------------------------------------------------


    
          

            local_dir = "/tmp/processed"  # Use /tmp in CodeBuild (only writable directory)
            os.makedirs(local_dir, exist_ok=True)
            
            # === Prepare DataFrame ===
            #trip_duration_df = df[['trip_duration']].head(100)
            
            # === Save CSV locally ===
            trip_prediction_file = os.path.join(local_dir, "trip_prediction.csv")
            pred_data.to_csv(trip_prediction_file, index=False, header=True)
            
            # === Upload to S3 ===
            s3 = boto3.resource('s3')
            trip_prediction_key = f"{s3_prefix}/output_trip_prediction/trip_prediction.csv"
            s3.Object(ml_s3_bucket, trip_prediction_key).upload_file(trip_prediction_file)
            
            print(f"Uploaded to: s3://{ml_s3_bucket}/{trip_prediction_key}")

           
           #------------------------------------------------------------------------------------------------
            # Convert to numeric, coercing errors to NaN
            pred_data[column_name] = pd.to_numeric(pred_data[column_name], errors='coerce')
            pred_na = pred_data[column_name].isna().sum()
            pred_clean = pred_data.dropna(subset=['prediction'])
            #return gt_df, na_count
        except Exception as e:
            print(f"Error loading column : {str(e)}")
            raise

        print(f"Loaded {len(pred_clean)} prediction records, {pred_na} invalid values")
        # Align datasets
        min_length = min(len(gt_clean), len(pred_clean))
        print(f"\nAligning datasets to minimum length: {min_length}")
        combined = pd.DataFrame({
            'ground_truth': gt_clean['ground_truth'].iloc[:min_length].reset_index(drop=True),
            'prediction': pred_clean['prediction'].iloc[:min_length].reset_index(drop=True)
        })
        # Data quality checks
        print("\nData quality summary:")
        print(f"Total rows before cleaning: {len(combined)}")
        print(f"Missing ground truth values: {combined['ground_truth'].isna().sum()}")
        print(f"Missing prediction values: {combined['prediction'].isna().sum()}")
        # Clean data
        #cleaned = combined.dropna()
        cleaned = combined
        print(f"\nRows remaining after cleaning: {len(cleaned)}")
        if len(cleaned) == 0:
            print("\nERROR: All data invalid after cleaning")
            print("Problematic data samples:")
            print("Ground truth with NaN values:")
            print(gt_data[gt_data['ground_truth'].isna()].head(3))
            print("\nPredictions with NaN values:")
            print(pred_data[pred_data['prediction'].isna()].head(3))
            raise ValueError("No valid data remaining after cleaning")
        # Save cleaned data
        baseline_file = "processed/baseline_data.csv"
        cleaned.to_csv(baseline_file, index=False, header=True)
        #print(f"\nSaved cleaned data to: {baseline_file}")
        # Upload to S3
        s3 = boto3.resource('s3')
        baseline_key = f"{s3_prefix}/monitoring/baseline_data.csv"
        s3.Object(ml_s3_bucket, baseline_key).upload_file(baseline_file)
        print(f"Uploaded baseline data to: s3://{ml_s3_bucket}/{baseline_key}")
        return f"s3://{ml_s3_bucket}/{baseline_key}"
    except Exception as e:
        print(f"\nDataset preparation failed: {str(e)}")
        print("Suggested troubleshooting steps:")
        print("1. Verify the input files exist in S3")
        print("2. Check that the CSV files contain the expected columns")
        print("3. Ensure numeric values are properly formatted")
        print("4. Validate there are matching prediction/ground truth pairs")
        raise
 
def main():
    try:
        sagemaker_session, region, _, role = initialize_resources()
        # Configuration
        #s3_bucket = "axcess-devst-sagemaker-bucket"
        #s3_prefix = "taxi-duration"
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Data preparation
        print("\n" + "="*50)
        print("Starting data preparation phase")
        baseline_uri = prepare_baseline_dataset(ml_s3_bucket, s3_prefix)
        print(f"\nBaseline dataset ready: {baseline_uri}")
        # Monitoring setup
        print("\n" + "="*50)
        print("Initializing ModelQualityMonitor")
        monitor = ModelQualityMonitor(
            role=role,
            sagemaker_session=sagemaker_session,
            instance_count=1,
            instance_type='ml.m5.2xlarge',
            volume_size_in_gb=50,
            max_runtime_in_seconds=3600
        )
        # Start baseline job
        job_name = f"model-quality-baseline-{current_time}"
        print(f"\nStarting monitoring job: {job_name}")
        baseline_job = monitor.suggest_baseline(
            job_name=job_name,
            baseline_dataset=baseline_uri,
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=f"s3://{ml_s3_bucket}/{s3_prefix}/monitoring/results/",
            problem_type='Regression',
            inference_attribute="prediction",
            ground_truth_attribute="ground_truth"
        )
        # Monitor execution
        print("\nJob execution started - waiting for completion...")
        baseline_job.wait(logs=True)
        # Show results
        #constraints = baseline_job.suggested_constraints()
        constraints = monitor.suggested_constraints()
        print("\n" + "="*50)
        print("Suggested Constraints:")
        print(pd.DataFrame.from_dict(constraints.body_dict["regression_constraints"], orient='index'))
        print("\nMonitoring setup completed successfully!")
    except Exception as e:
        print(f"\nWorkflow failed: {str(e)}")
        print("\nTroubleshooting guide:")
        print("1. Check the error message above")
        print("2. Verify S3 file paths and permissions")
        print("3. Examine the input data format")
        print("4. Confirm IAM role has required permissions")
        print("5. Check CloudWatch logs for detailed errors")
        raise
 
if __name__ == "__main__":
    main()
