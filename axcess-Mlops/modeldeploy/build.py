import argparse
import boto3
from botocore.exceptions import ClientError
import json
import logging
import os
from modeldeploy.pipelines import run_pipeline
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.model import XGBoostModel
from sagemaker.model import Model
import traceback

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
print("BASE_DIR: ", BASE_DIR)

logger = logging.getLogger(__name__)
sagemaker_client = boto3.client("sagemaker")
sagemaker_session = sagemaker.Session()


def get_approved_package(model_package_group_name):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """
    try:
        # Get the latest approved model package
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = (
                "No approved ModelPackage found for ModelPackageGroup: {}".format(model_package_group_name))
            print("{}".format(error_message))

            raise Exception(error_message)

        model_package = approved_packages[0]
        print("Identified the latest approved model package: {}".format(model_package))

        return model_package
    except ClientError as e:
        stacktrace = traceback.format_exc()
        error_message = e.response["Error"]["Message"]
        print("{}".format(stacktrace))

        raise Exception(error_message)


def describe_model_package(model_package_arn):
    try:
        model_package = sagemaker_client.describe_model_package(
            ModelPackageName=model_package_arn
        )

        print("{}".format(model_package))

        if len(model_package) == 0:
            error_message = ("No ModelPackage found for: {}".format(model_package_arn))
            print("{}".format(error_message))

            raise Exception(error_message)

        return model_package
    except ClientError as e:
        stacktrace = traceback.format_exc()
        error_message = e.response["Error"]["Message"]
        print("{}".format(stacktrace))

        raise Exception(error_message)


def extend_config(args, pipeline_definitions, container_definitions, stage_config, input_path,output_path,use_case):
    """
    Extend the stage configuration with additional parameters and tags based.
    """
    # Verify that config has parameters and tags sections
    if not "Parameters" in stage_config or not "StageName" in stage_config["Parameters"]:
        raise Exception("Configuration file must include SageName parameter")
    if not "Tags" in stage_config:
        stage_config["Tags"] = {}
    # Create new params and tags
    new_params = {
        "InputPath": input_path,
        "OutputPath": output_path,
        "SageMakerProjectName": args.sagemaker_project_name,
        "SageMakerProjectId": args.sagemaker_project_id,
        "ModelExecutionRoleArn": args.model_execution_role,
        #"Tenant": tenant,
        #"Algorithm": algorithm,
        "UseCase": use_case

    }

    # tenant = tenant
    # algorithm = algorithm
    # use_case = use_case

    # index = 1
    for pipeline_definition in pipeline_definitions:
        new_params["PipelineDefinitionBody"] = pipeline_definition
        # index += 1

    # index = 1
    for container_def in container_definitions:
        new_params["ContainerImage"] = container_def["Image"]
        new_params["ModelDataUrl"] = container_def["ModelDataUrl"]
        new_params["ModelName"] = container_def["ModelName"]
        # index += 1
        print('new_params_ModelDataUrl:',new_params["ModelDataUrl"])

    new_tags = {
        "sagemaker:deployment-stage": stage_config["Parameters"]["StageName"],
        "sagemaker:project-id": args.sagemaker_project_id,
        "sagemaker:project-name": args.sagemaker_project_name,
    }
    # Add tags from Project
    get_pipeline_custom_tags(args, sagemaker_client, new_tags)

    return {
        "Parameters": {**stage_config["Parameters"], **new_params},
        "Tags": {**stage_config.get("Tags", {}), **new_tags},
    }


def get_pipeline_custom_tags(args, sagemaker_client, new_tags):
    try:
        response = sagemaker_client.list_tags(
            ResourceArn=args.sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags[project_tag["Key"]] = project_tag["Value"]
    except:
        logger.error("Error getting project tags")
    return new_tags


def get_cfn_style_config(stage_config):
    parameters = []
    for key, value in stage_config["Parameters"].items():
        parameter = {
            "ParameterKey": key,
            "ParameterValue": value
        }
        parameters.append(parameter)
    tags = []
    for key, value in stage_config["Tags"].items():
        tag = {
            "Key": key,
            "Value": value
        }
        tags.append(tag)
    return parameters, tags


def create_cfn_params_tags_file(config, export_params_file, export_tags_file):
    # Write Params and tags in separate file for Cfn cli command
    parameters, tags = get_cfn_style_config(config)
    with open(export_params_file, "w") as f:
        json.dump(parameters, f, indent=4)
    with open(export_tags_file, "w") as f:
        json.dump(tags, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--aws-region", type=str, required=True)
    parser.add_argument("--default-bucket", type=str, required=True)
    parser.add_argument("--model-execution-role", type=str, required=True)
    parser.add_argument("--model-package-group-name", type=str, required=True)
    parser.add_argument("--sagemaker-project-id", type=str, required=True)
    parser.add_argument("--sagemaker-project-name", type=str, required=True)
    parser.add_argument("--inference-framework-version", type=str, required=True)
    parser.add_argument("--inference-python-version", type=str, required=True)
    parser.add_argument("--sagemaker-project-arn", type=str, required=False)
    parser.add_argument("--import-staging-config", type=str, default="staging-config.json")
    parser.add_argument("--import-prod-config", type=str, default="prod-config.json")
    parser.add_argument("--export-staging-config", type=str, default="staging-config-export.json")
    parser.add_argument("--export-staging-params", type=str, default="staging-params-export.json")
    parser.add_argument("--export-staging-tags", type=str, default="staging-tags-export.json")
    parser.add_argument("--export-prod-config", type=str, default="prod-config-export.json")
    parser.add_argument("--export-prod-params", type=str, default="prod-params-export.json")
    parser.add_argument("--export-prod-tags", type=str, default="prod-tags-export.json")
    parser.add_argument("--export-cfn-params-tags", type=bool, default=False)
    parser.add_argument("--inference-instance-type", type=str, default="ml.m5.large")
    parser.add_argument("--inference-instance-count", type=str, default=1)
    parser.add_argument("--kwargs", type=str, default=None)
    args, _ = parser.parse_known_args()

    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=args.log_level)

    model_names = []
    pipeline_definitions = []
    container_definitions = []
    inference_image_uri = '317185619046.dkr.ecr.us-east-1.amazonaws.com/inference-repo:41.0'
    parsed_kwargs = json.loads(args.kwargs)
    model_package_group_name = args.model_package_group_name
    default_bucket = args.default_bucket
    s3_prefix = parsed_kwargs.get('s3_prefix')
    #tenant = parsed_kwargs.get('tenant')
    #algorithm = parsed_kwargs.get('algorithm')
    use_case = parsed_kwargs.get('use_case')
    model_package_group_name_input = parsed_kwargs.get('model_package_group_name_input')
    model_version_input = parsed_kwargs.get('model_version_input')
    model_name = f"{model_package_group_name_input}-{model_version_input}"
    #model_package_group_name = args.model_package_group_name
    print('model_package_group_name:',model_package_group_name)
    print("Generated model_name from input variables:",model_name)
    # args.default_bucket = 'xfactrs-devst-sagemaker'
    # prefix = 'classification'


        # creating s3 path 
    
    if s3_prefix != "NA": 
        input_path = f"s3://{default_bucket}/{s3_prefix}/{use_case}/data/test/input"
        output_path = f"s3://{default_bucket}/{s3_prefix}/{use_case}/data/test/output"
    else:
        #input_path = f"s3://{default_bucket}/{use_case}/data/test/input"
        #input_path = "s3://axcess-devst-sagemaker-bucket/taxi-duration/test"
        input_path = "s3://axcess-devst-sagemaker-bucket/taxi-duration/batch_input"
        #output_path = "s3://axcess-devst-sagemaker-bucket/taxi-duration/test/data/output/"
        #input_path = "s3://axcess-devst-sagemaker-bucket/taxi-duration/batch_input/inference_data.csv"
        output_path = "s3://axcess-devst-sagemaker-bucket/taxi-duration/batch_output/"


    # for model_package_group_name in args.model_package_group_names.split(","):
    logger.info("Model Package Group: {}".format(model_package_group_name))
    # Get the latest approved package
    model_package = get_approved_package(model_package_group_name)
    model_package_arn = model_package["ModelPackageArn"]
    model_package = describe_model_package(model_package_arn)
    
    print('model_package:',model_package)

    # Getting ModelPackageGroupName and ModelPackageVersion
    ModelPackageGroupName = model_package["ModelPackageGroupName"]
    ModelPackageVersion = model_package["ModelPackageVersion"]
    Model_Name = f"{ModelPackageGroupName}-{ModelPackageVersion}"
    print("ModelPackageGroupName:",ModelPackageGroupName)
    print("ModelPackageVersion:",ModelPackageVersion)
    print("Model_Name:",Model_Name)
    
    print("The folder structure is: ")
    for path, subdirs, files in os.walk(BASE_DIR):
        for name in files:
            print(os.path.join(path, name))
            
    print('source_dir: ', os.path.join(".",BASE_DIR, parsed_kwargs["use_case"],"mlops"))

    # Checking the input for model_name is similar to model_name created in above steps
    if model_name == Model_Name:
        
        model = Model(
            image_uri=inference_image_uri,
            source_dir=os.path.join(".",BASE_DIR, parsed_kwargs["use_case"],"mlops"),
            entry_point="inference.py",
            name=model_package_group_name + "-" + str(model_package["ModelPackageVersion"]),
            # framework_version=str(args.inference_framework_version),
            # py_version=args.inference_python_version,
            model_data=model_package["InferenceSpecification"]["Containers"][0]["ModelDataUrl"],
            role=args.model_execution_role,
            sagemaker_session=sagemaker_session
        )
        
        print("Model name from the input is matching with generated model name from above steps.")
        print('model:',model)
        print('model_data:',model_package["InferenceSpecification"]["Containers"][0]["ModelDataUrl"])
    else:
        print("Model name from the input is not matching with generated model name from above steps. Please check Model Registry Group and provide correct model version.")

    container_def = model.prepare_container_def(instance_type=args.inference_instance_type)
    container_def["ModelName"] = model_package_group_name + "-" + str(model_package["ModelPackageVersion"])
    
    # check modeldataurl from model_package and container_def are same or not
#        if model_package["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] != container_def['ModelDataUrl']:
#            container_def['ModelDataUrl'] = model_package["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
        
        
    print('container_def:',container_def)

    container_definitions.append(container_def)

    model_names.append(model_package_group_name + "-" + str(model_package["ModelPackageVersion"]))

    # Build the pipeline
    pipeline_definition = run_pipeline.main(
        'trip-prediction.mlops.pipeline_deploy',
        args.model_execution_role,
        json.dumps([
            {"Key": "sagemaker:project-name", "Value": args.sagemaker_project_name},
            {"Key": "sagemaker:project-id", "Value": args.sagemaker_project_id}
        ]),
        json.dumps({
            'region': args.aws_region,
            'default_bucket': args.default_bucket,
            'model_names': model_names,
            'inference_instance_type': args.inference_instance_type,
            'inference_instance_count': args.inference_instance_count
        })
    )

    pipeline_definitions.append(pipeline_definition)
    
    print('pipeline_definitions:',pipeline_definitions)
    print('container_definitions:',container_definitions)

    # Write the staging config
    staging_config_file = os.path.join("/",BASE_DIR, "modeldeploy", args.import_staging_config)
    staging_config_file_export = os.path.join("/",BASE_DIR, "modeldeploy", args.export_staging_config)
    print("staging_config_file: ", staging_config_file)
    print("args.import_staging_config: ", args.import_staging_config)
    # with open(f"modeldeploy/{args.import_staging_config}", "r") as f:
    with open(staging_config_file, "r") as f:
        staging_config = extend_config(args, pipeline_definitions, container_definitions, json.load(f),input_path,output_path,use_case)
    logger.debug("Staging config: {}".format(json.dumps(staging_config, indent=4)))
    with open(staging_config_file_export, "w") as f:
        json.dump(staging_config, f, indent=4)
    if (args.export_cfn_params_tags):
        create_cfn_params_tags_file(staging_config, args.export_staging_params, args.export_staging_tags)

    # Write the prod config for code pipeline
    prod_config_file = os.path.join("/",BASE_DIR, "modeldeploy", args.import_prod_config)
    prod_config_file_export = os.path.join("/",BASE_DIR, "modeldeploy", args.export_prod_config)
    print("prod_config_file: ", prod_config_file)
    print("args.import_prod_config: ", args.import_prod_config)
    with open(prod_config_file, "r") as f:
        prod_config = extend_config(args, pipeline_definitions, container_definitions, json.load(f),input_path,output_path,use_case)
    logger.debug("Prod config: {}".format(json.dumps(prod_config, indent=4)))
    with open(prod_config_file_export, "w") as f:
        json.dump(prod_config, f, indent=4)
    if (args.export_cfn_params_tags):
        create_cfn_params_tags_file(prod_config, args.export_prod_params, args.export_prod_tags)

    print("The folder structure is: ")
    for path, subdirs, files in os.walk(BASE_DIR):
        for name in files:
            print(os.path.join(path, name))


if __name__ == "__main__":
    main()
