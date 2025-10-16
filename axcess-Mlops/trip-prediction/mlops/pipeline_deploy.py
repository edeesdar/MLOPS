import os
import boto3
import sagemaker
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TransformStep, Transformer, TransformInput

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

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn.lower())
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags

def get_pipeline(
    region,
    model_names,
    role=None,
    default_bucket=None,
    sagemaker_project_arn=None,
    pipeline_name="TaxiTripDurationBatchPipeline",
    inference_instance_type="ml.m5.xlarge",
    inference_instance_count=1
):

    sagemaker_session = get_session(region, default_bucket)

    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    print('role:',role)
    
    pipeline_session = get_pipeline_session(region, default_bucket)
    #sagemaker_session = get_session(region, default_bucket)
    
    
    
    # Parameters
    input_path = ParameterString("InputPath")
    output_path = ParameterString("OutputPath")

    input_path_new = "s3://mlops-workshop-edees-nasrullah-dar/taxi-duration/batch_input/inference_data.csv" 
    output_path_new = "s3://mlops-workshop-edees-nasrullah-dar/taxi-duration/batch_output/inference_data.csv.out"
    

    #input_path = "s3://axcess-devst-sagemaker-bucket/taxi-duration/batch_input/"
    #output_path = "s3://axcess-devst-sagemaker-bucket/taxi-duration/batch_output/"

    for model_name in model_names:
    
    
        # Transformer configuration
        transformer = Transformer(
            model_name=model_name,
            instance_count=inference_instance_count,
            instance_type=inference_instance_type,
            output_path=output_path,
            accept="text/csv",
            strategy="MultiRecord"
        )
        
        # Transform Step
        transform_step = TransformStep(
            name="BatchPredictTripDuration",
            transformer=transformer,
            inputs=TransformInput(
                data=input_path,
                content_type="text/csv",
                split_type="Line"
            )
        )
    
    # Pipeline Definition
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[input_path, output_path],
        steps=[transform_step],
        sagemaker_session=sagemaker_session
    )
    
    return pipeline

