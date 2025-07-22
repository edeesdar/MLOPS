import os
import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.estimator import Estimator

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
Source_dir = os.path.abspath("workspace/mlops")

def get_sagemaker_client(region):
    """Gets the sagemaker client.

       Args:
           region: the aws region to start the session
           default_bucket: the bucket to use for storing the artifacts

       Returns:
           `sagemaker.session.Session instance
       """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client

def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline(
    region,
    role=None,
    use_case=None,
    default_bucket=None,
    sagemaker_project_arn=None,
    pipeline_name="TaxiTripDurationPipeline",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
    inference_instance_type="ml.m5.xlarge",
    ml_s3_bucket="axcess-devst-sagemaker-bucket",
    base_job_prefix=None,
    s3_prefix="NA",
    model_package_group_name="TaxiTripDurationPackageGroup",
    processing_instance_count_param= "1",
    training_instance_count_param= "1"
    
):
    pipeline_session = get_pipeline_session(region, default_bucket)
    
    if role is None:
        role = sagemaker.session.get_execution_role(pipeline_session)
    
    # Parameters
    input_data = ParameterString(
        name="InputData",
        default_value="s3://axcess-devst-sagemaker-bucket-3/taxi-duration/original_raw_data/data.csv"
    )
    
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval"
    )
    
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount",
        default_value=1
    )
    
    training_instance_count = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=1
    )

    # processing last demo ready success : '345594592951.dkr.ecr.us-east-1.amazonaws.com/processing-repo:10.0
    # Processing Step
    processor = ScriptProcessor(
        command=['python3'],
        image_uri='345594592951.dkr.ecr.us-east-1.amazonaws.com/processing-repo:13.0',
        role=role,
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        sagemaker_session=pipeline_session
    )
    
    run_args = processor.get_run_args(
        code=os.path.join(BASE_DIR, "processing.py"),
        #inputs=[
        #    ProcessingInput(
        #        input_name="input_data",
        #        source=input_data,
        #        destination="/opt/ml/processing/input"
        #    )
        #],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{ml_s3_bucket}/taxi-duration/train"
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation",
                destination=f"s3://{ml_s3_bucket}/taxi-duration/validation"
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/inference",
                destination=f"s3://{ml_s3_bucket}/taxi-duration/test"
            )
        ]
    )
    
    step_process = ProcessingStep(
        name="PreprocessTaxiData",
        processor=processor,
        #inputs=run_args.inputs,
        outputs=run_args.outputs,
        code=run_args.code
    )
    
    # Training Step
    hyperparameters = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "gamma": 0,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0,
        "reg_lambda": 1
    }

    # training last demo ready success : '345594592951.dkr.ecr.us-east-1.amazonaws.com/training-repo:1.0'
    estimator = Estimator(
        image_uri='345594592951.dkr.ecr.us-east-1.amazonaws.com/training-repo:8.0',
        entry_point="train.py",
        source_dir=BASE_DIR,
        role=role,
        instance_count=training_instance_count,
        instance_type=training_instance_type,
        output_path=f"s3://{default_bucket}/taxi-duration/model",
        hyperparameters=hyperparameters,
        sagemaker_session=pipeline_session
    )
    
    step_train = TrainingStep(
        name="TrainTripDurationModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )
    
    # Model Registration Step
    step_register = RegisterModel(
        name="RegisterTripDurationModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=[inference_instance_type],
        transform_instances=[inference_instance_type],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status
    )
    
    # Pipeline Definition
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data,
            model_approval_status,
            processing_instance_count,
            training_instance_count
        ],
        steps=[step_process, step_train, step_register],
        sagemaker_session=pipeline_session
    )
    
    return pipeline
