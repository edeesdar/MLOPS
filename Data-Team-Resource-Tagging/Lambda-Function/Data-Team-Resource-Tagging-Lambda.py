import json
import boto3
import os
from datetime import datetime, timedelta

# ---------------------------------- main lambda handler -----------------------------------------#

def lambda_handler(event, context):
    detail = event
    print('event:',event)
    source = detail.get('source', '')
    print('source:',source)
    
    # Extract user email from event
    user_email = extract_user_email_from_event(detail)
    print('user_email:',user_email)
    
    # Get resource ARN from EventBridge event
    resource_arn = get_resource_from_event(detail)
    print('resource_arn:',resource_arn)
    
    if not resource_arn:
        return {'statusCode': 200, 'body': 'No resource found'}
    
    # Check if sponsor=Pratik tag exists
    has_sponsor_tag = check_sponsor_tag(detail,resource_arn,user_email)
    
    if not has_sponsor_tag:
        # Send email notification
        send_notification(user_email, resource_arn, source)
        
        # Add required tags
        add_required_tags(detail,resource_arn, user_email)
    
    return {'statusCode': 200, 'body': 'Tag check completed'}

# ---------------------------------- extract users email-id from event ------------------------------------- #

def extract_user_email_from_event(detail):
    """Extract email from EventBridge event"""
    user_identity = detail.get('detail', {}).get('userIdentity', {})
    user_arn = user_identity.get('arn', '')
    
    if 'assumed-role' in user_arn and 'AWSReservedSSO' in user_arn:
        return user_arn.split('/')[-1]
    return user_arn.split('/')[-1]

# ------------------------------------ get resource arn or name from event ------------------------ ---------- #

def get_resource_from_event(detail):
    """Extract resource ARN directly from EventBridge event"""
    source = detail.get('source', '')
    event_name = detail.get('detail', {}).get('eventName', {})
    
    if source == 'aws.s3':
        return f"arn:aws:s3:::{detail.get('detail', {}).get('requestParameters', {}).get('bucketName', '')}"
    elif source == 'aws.ec2' and event_name == 'RunInstances':
        return detail.get('detail', {}).get('responseElements', {}).get('instancesSet',{}).get('items',{})[0].get('instanceId',{})
    elif source == 'aws.ec2' and event_name == 'CreateImage':
        return detail.get('detail', {}).get('responseElements', {}).get('imageId',{})
    # Volume
    elif source == 'aws.ec2' and event_name == 'CreateVolume':
        return detail.get('detail', {}).get('responseElements', {}).get('volumeId',{})
    elif source == 'aws.rds':
        return detail.get('detail', {}).get('source-id', '')
    elif source == 'aws.lambda':
        return detail.get('detail', {}).get('responseElements', {}).get('functionArn', '')

    # Domain
    elif source == 'aws.sagemaker' and event_name == 'CreateDomain':
        return detail.get('detail', {}).get('responseElements', {}).get('domainArn', '')

    elif source == 'aws.sagemaker' and event_name == 'CreateUserProfile':
        return detail.get('detail', {}).get('responseElements', {}).get('userProfileArn', '')

    elif source == 'aws.sagemaker' and event_name == 'CreateModel':
        return detail.get('detail', {}).get('responseElements', {}).get('modelArn', '')

    # Glue
    elif source == 'aws.glue' and event_name == 'CreateJob':
        return detail.get('detail', {}).get('responseElements', {}).get('name', '')

    elif source == 'aws.glue' and event_name == 'CreateDatabase':
        return detail.get('detail', {}).get('requestParameters', {}).get('databaseInput',{}).get('name',{})
    
    elif source == 'aws.glue' and event_name == 'CreateTable':
        return detail.get('detail', {}).get('requestParameters', {}).get('tableInput',{}).get('name',{})

    elif source == 'aws.glue' and event_name == 'CreateCrawler':
        return detail.get('detail',{}).get('requestParameters',{}).get('name',{})

    # RDS
    elif source == 'aws.rds' and event_name == 'CreateDBInstance':
        return detail.get('detail', {}).get('responseElements',{}).get('dBInstanceIdentifier')

    elif source == 'aws.rds' and event_name == 'CreateDBSnapshot':
        return detail.get('detail',{}).get('responseElements',{}).get('dBSnapshotArn',{})

    # Lambda
    elif source == 'aws.lambda' and event_name == 'CreateFunction20150331':
        return detail.get('detail',{}).get('requestParameters',{}).get('functionName',{})

    # Kafka
    elif source == 'aws.kafka' and event_name == 'CreateCluster':
        return detail.get('detail',{}).get("responseElements", {}).get("clusterArn", {})

    # ServiceCatalog
    elif source == 'aws.servicecatalog' and event_name == 'CreatePortfolio':
        return detail.get('detail',{}).get('responseElements',{}).get('portfolioDetail',{}).get('id',{})

    elif source == 'aws.servicecatalog' and event_name == 'CreateProduct':
        return detail.get('detail',{}).get('requestParameters',{}).get('name',{})

    # DMS

    elif source == 'aws.dms' and event_name == 'CreateReplicationInstance':
        return detail.get("detail", {}).get("responseElements", {}).get("replicationInstance", {}).get("replicationInstanceArn", {})

    elif source == 'aws.dms' and event_name == 'CreateEndpoint':
        return detail.get("detail", {}).get("responseElements", {}).get("endpoint", {}).get("endpointArn", {})

    elif source == 'aws.dms' and event_name == 'CreateReplicationTask':
        return detail.get("detail", {}).get("responseElements", {}).get("replicationTask", {}).get("replicationTaskArn", {})

    # Codepipeline

    elif source == 'aws.codepipeline' and event_name == 'CreatePipeline':
        return detail.get("detail", {}).get("responseElements", {}).get("pipeline", {}).get("name", {})

    # CodeBuild: Tag api's not available.
    
    # CloudFormation: Tag api's not available.

    # EKS

    elif source == 'aws.' and event_name == 'CreatePipeline':
        return detail.get("detail", {}).get("responseElements", {}).get("pipeline", {}).get("name", {})

    # IAM

    elif source == 'aws.iam' and event_name == 'CreateRole':
        return detail.get("detail", {}).get("requestParameters", {}).get("roleName", {})

    elif source == 'aws.iam' and event_name == 'CreatePolicy':
        return detail.get("detail", {}).get("responseElements", {}).get("policy", {}).get("arn", {})

    # EventBridge

    elif source == 'aws.events' and event_name == 'CreateEventBus':
        return detail.get("detail", {}).get("responseElements", {}).get("eventBusArn", {})

    elif source == 'aws.events' and event_name == 'PutRule': # create rule
        return detail.get("detail", {}).get("responseElements", {}).get("ruleArn", {})

    # VPC
    elif source == 'aws.ec2' and event_name == 'CreateVpc':
        return detail.get("detail", {}).get("responseElements", {}).get("vpc", {}).get("vpcId", {})

    # Subnet
    elif source == 'aws.ec2' and event_name == 'CreateSubnet':
        return detail.get("detail", {}).get("responseElements", {}).get("subnet", {}).get("subnetId", {})
   
    return None

# ------------------------------------- check sponsor tag is already exist ---------------------------------- #


def check_sponsor_tag(detail,resource_arn,user_email):
    """Check if resource has sponsor=Pratik tag"""
    print('detail:',detail)
    source = detail.get('source', '')
    service = source.split('.')[-1]
    print('service:',service)
    event_name = detail.get('detail', {}).get('eventName', {})
    account_id = detail["account"]
    region = detail["region"]

    print('account_id:',account_id)
    print('region:',region)
    
    try:
        # S3
        if service == 's3':
            s3 = boto3.client('s3')
            bucket = resource_arn.split(':')[-1]
            response = s3.get_bucket_tagging(Bucket=bucket)
            tags = {tag['Key']: tag['Value'] for tag in response.get('TagSet', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email
        
        # EC2
        elif service == 'ec2' and event_name == 'RunInstances':
            ec2 = boto3.client('ec2')
            resource_id = resource_arn
            response = ec2.describe_tags(
                Filters=[
                    {'Name': 'resource-id', 'Values': [resource_id]},
                    {'Name': 'key', 'Values': ['sponsor']},
                    {'Name': 'value', 'Values': ['Pratik']}
                ]
            )
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email
            # return len(response.get('Tags', [])) > 0
        # EC2 : Image
        elif service == 'ec2' and event_name == 'CreateImage':
            ec2 = boto3.client('ec2')
            resource_id = resource_arn
            response = ec2.describe_tags(
                Filters=[
                    {'Name': 'resource-id', 'Values': [resource_id]},
                    {'Name': 'key', 'Values': ['sponsor']},
                    {'Name': 'value', 'Values': ['Pratik']}
                ]
            )
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email
            # return len(response.get('Tags', [])) > 0
        # EC2 : Volume
        elif service == 'ec2' and event_name == 'CreateVolume':
            ec2 = boto3.client('ec2')
            resource_id = resource_arn
            response = ec2.describe_tags(
                Filters=[
                    {'Name': 'resource-id', 'Values': [resource_id]},
                    {'Name': 'key', 'Values': ['sponsor']},
                    {'Name': 'value', 'Values': ['Pratik']}
                ]
            )
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email
            # return len(response.get('Tags', [])) > 0
        
        elif service == 'rds':
            rds = boto3.client('rds')
            response = rds.list_tags_for_resource(ResourceName=resource_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('TagList', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email
        
        elif service == 'lambda':
            lambda_client = boto3.client('lambda')
            response = lambda_client.list_tags(Resource=resource_arn)
            tags = response.get('Tags', {})
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email
        
        elif service == 'sagemaker' and event_name == 'CreateDomain':
            sagemaker = boto3.client('sagemaker')
            response = sagemaker.list_tags(ResourceArn=resource_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        elif service == 'sagemaker' and event_name == 'CreateUserProfile':
            sagemaker = boto3.client('sagemaker')
            response = sagemaker.list_tags(ResourceArn=resource_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        # Glue
        elif service == 'glue' and event_name == 'CreateJob':
            glue = boto3.client('glue')
            response = glue.get_tags(ResourceArn=f"arn:aws:glue:{region}:{account_id}:job/{resource_arn}")
            tags = response.get('Tags', {})
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email
        elif service == 'glue' and event_name == 'CreateDatabase':
            glue = boto3.client('glue')
            response = glue.get_tags(ResourceArn=f"arn:aws:glue:{region}:{account_id}:database/{resource_arn}")
            tags = response.get('Tags', {})
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email
        elif service == 'glue' and event_name == 'CreateTable':
            db_name = detail.get('detail', {}).get('requestParameters', {}).get('databaseName',{})
            glue = boto3.client('glue')
            response = glue.get_tags(ResourceArn=f"arn:aws:glue:{region}:{account_id}:table/{db_name}/{resource_arn}")
            tags = response.get('Tags', {})
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email
        elif service == 'glue' and event_name == 'CreateCrawler':
            glue = boto3.client('glue')
            response = glue.get_tags(ResourceArn=f"arn:aws:glue:{region}:{account_id}:crawler/{resource_arn}")
            tags = response.get('Tags', {})
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        # RDS
        # rds
        elif service == 'rds' and event_name == 'CreateDBInstance':
            rds = boto3.client('rds')
            arn = f"arn:aws:rds:{region}:{account_id}:db:{resource_arn}"
            response = rds.list_tags_for_resource(ResourceName=arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        elif service == 'rds' and event_name == 'CreateDBSnapshot':
            rds = boto3.client('rds')
            #arn = f"arn:aws:rds:{region}:{account_id}:db:{resource_arn}"
            response = rds.list_tags_for_resource(ResourceName=resource_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        # Lambda
        
        elif service == 'lambda' and event_name == 'CreateFunction20150331':
            lambda_client = boto3.client('lambda')
            arn = f"arn:aws:lambda:{region}:{account_id}:function:{resource_arn}"
            response = lambda_client.list_tags(Resource=arn)
            tags = response.get('Tags', {})
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        # Kafka
        elif source == 'aws.kafka' and event_name == 'CreateCluster':
            kafka_client = boto3.client("kafka")
            response = kafka_client.list_tags_for_resource(ResourceArn=resource_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        # ServiceCatalog
        elif source == 'aws.servicecatalog' and event_name == 'CreatePortfolio':
            client = boto3.client('servicecatalog')
            response = client.describe_portfolio(Id=resource_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        elif source == 'aws.servicecatalog' and event_name == 'CreateProduct':
            client = boto3.client('servicecatalog')
            response = client.describe_product_as_admin(Name=resource_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        # DMS
        elif source == 'aws.dms' and event_name == 'CreateReplicationInstance':
            dms = boto3.client("dms")
            response = dms.list_tags_for_resource(ResourceArn=resource_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        elif source == 'aws.dms' and event_name == 'CreateEndpoint':
            dms = boto3.client("dms")
            response = dms.list_tags_for_resource(ResourceArn=resource_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        elif source == 'aws.dms' and event_name == 'CreateReplicationTask':
            dms = boto3.client("dms")
            response = dms.list_tags_for_resource(ResourceArn=resource_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        # Codepipeline
        elif source == 'aws.codepipeline' and event_name == 'CreatePipeline':
            codepipeline_client = boto3.client('codepipeline')
            pipliene_arn = f'arn:aws:codepipeline:{region}:{account_id}:{resource_arn}'
            response = codepipeline_client.list_tags_for_resource(resourceArn=pipliene_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        # IAM
        elif source == 'aws.iam' and event_name == 'CreateRole':
            iam_client = boto3.client('iam')
            response = iam_client.list_role_tags(RoleName=resource_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        elif source == 'aws.iam' and event_name == 'CreatePolicy':
            iam_client = boto3.client('iam')
            response = iam_client.list_policy_tags(PolicyArn=resource_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        # EventBridge
        elif source == 'aws.events' and event_name == 'CreateEventBus':
            event_client = boto3.client('events')
            response = event_client.list_tags_for_resource(ResourceARN=resource_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        elif source == 'aws.events' and event_name == 'PutRule': # create rule
            event_client = boto3.client('events')
            response = event_client.list_tags_for_resource(ResourceARN=resource_arn)
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        # VPC
        elif service == 'ec2' and event_name == 'CreateVpc':
            ec2 = boto3.client('ec2')
            resource_id = resource_arn
            response = ec2.describe_tags(
                Filters=[
                    {'Name': 'resource-id', 'Values': [resource_id]}
                ]
            )
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email

        # Subnet
        elif service == 'ec2' and event_name == 'CreateSubnet':
            ec2 = boto3.client('ec2')
            resource_id = resource_arn
            response = ec2.describe_tags(
                Filters=[
                    {'Name': 'resource-id', 'Values': [resource_id]}
                ]
            )
            tags = {tag['Key']: tag['Value'] for tag in response.get('Tags', [])}
            return tags.get('sponsor') == 'Pratik' and tags.get('user') == user_email
        
    
    except Exception as e:
        print(f"Error checking {service} tags: {e}")
        return False
    
    return False

# ------------------------- send notification via email to administrator and executor ------------------------ #


def send_notification(user_email, resource_arn, source):
    """Send SNS notification"""
    sns = boto3.client('sns')
    
    
    message = f"""Missing Required Tag Alert

Resource: {resource_arn}
Service: {source}
User: {user_email}

The resource was created without the required tags.
Required tags have been automatically added.

Required tags:
- sponsor: Pratik
- user: {user_email}
"""
    
    sns.publish(
        TopicArn=os.environ['SNS_TOPIC_ARN'],
        Subject="Missing Required Tags - Action Required",
        Message=message
    )

# ----------------------------- add required tags to resources if missing ------------------------------ #


def add_required_tags(detail,resource_arn, user_email):
    """Add sponsor=Pratik and user=email tags"""
    from datetime import datetime, timedelta, timezone

    ist = timezone(timedelta(hours=5, minutes=30))
    current_time_ist = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    
    source = detail.get('source', '')
    service = source.split('.')[-1]
    event_name = detail.get('detail', {}).get('eventName', {})
    print('service:',service)
    tags = {'sponsor': 'Pratik', 'user': user_email,'ResourceCreationDate':current_time_ist}

    account_id = detail["account"]
    region = detail["region"]
    
    try:
        # s3
        if service == 's3':
            s3 = boto3.client('s3')
            bucket = resource_arn.split(':')[-1]
            try:
                response = s3.get_bucket_tagging(Bucket=bucket)
                existing_tags = {tag['Key']: tag['Value'] for tag in response.get('TagSet', [])}
                existing_tags.update(tags)
                tags = existing_tags
            except:
                pass
            tag_set = [{'Key': k, 'Value': v} for k, v in tags.items()]
            s3.put_bucket_tagging(Bucket=bucket, Tagging={'TagSet': tag_set})
        
        # EC2
        elif service == 'ec2' and event_name == 'RunInstances':
            ec2 = boto3.client('ec2')
            resource_id = resource_arn
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            ec2.create_tags(Resources=[resource_id], Tags=tag_list)
        # EC2: Image
        elif service == 'ec2' and event_name == 'CreateImage':
            ec2 = boto3.client('ec2')
            resource_id = resource_arn
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            ec2.create_tags(Resources=[resource_id], Tags=tag_list)
        # EC2: Volume
        elif service == 'ec2' and event_name == 'CreateVolume':
            ec2 = boto3.client('ec2')
            resource_id = resource_arn
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            ec2.create_tags(Resources=[resource_id], Tags=tag_list)
        
        elif service == 'rds':
            rds = boto3.client('rds')
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            rds.add_tags_to_resource(ResourceName=resource_arn, Tags=tag_list)
        
        elif service == 'lambda':
            lambda_client = boto3.client('lambda')
            lambda_client.tag_resource(Resource=resource_arn, Tags=tags)
        
        elif service == 'sagemaker' and event_name == 'CreateDomain':
            sagemaker = boto3.client('sagemaker')
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            sagemaker.add_tags(ResourceArn=resource_arn, Tags=tag_list)

        elif service == 'sagemaker' and event_name == 'CreateUserProfile':
            sagemaker = boto3.client('sagemaker')
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            sagemaker.add_tags(ResourceArn=resource_arn, Tags=tag_list)

        # Glue
        elif service == 'glue' and event_name == 'CreateJob':
            glue = boto3.client('glue')
            #tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            glue.tag_resource(ResourceArn=f"arn:aws:glue:{region}:{account_id}:job/{resource_arn}",TagsToAdd=tags)

        elif service == 'glue' and event_name == 'CreateDatabase':
            glue = boto3.client('glue')
            #tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            glue.tag_resource(ResourceArn=f"arn:aws:glue:{region}:{account_id}:database/{resource_arn}",TagsToAdd=tags)

        elif service == 'glue' and event_name == 'CreateTable':
            glue = boto3.client('glue')
            db_name = detail.get('detail', {}).get('requestParameters', {}).get('databaseName',{})
            #tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            glue.tag_resource(ResourceArn=f"arn:aws:glue:{region}:{account_id}:table/{db_name}/{resource_arn}",TagsToAdd=tags)

        elif service == 'glue' and event_name == 'CreateCrawler':
            glue = boto3.client('glue')
            #tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            glue.tag_resource(ResourceArn=f"arn:aws:glue:{region}:{account_id}:crawler/{resource_arn}",TagsToAdd=tags)
         
        # RDS
        elif service == 'rds' and event_name == 'CreateDBInstance':
            rds = boto3.client('rds')
            arn = f"arn:aws:rds:{region}:{account_id}:db:{resource_arn}"
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            rds.add_tags_to_resource(ResourceName=arn,Tags=tag_list)

        elif service == 'rds' and event_name == 'CreateDBSnapshot':
            rds = boto3.client('rds')
            #arn = f"arn:aws:rds:{region}:{account_id}:db:{resource_arn}"
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            rds.add_tags_to_resource(ResourceName=resource_arn,Tags=tag_list)

        # Lambda
        elif service == 'lambda' and event_name == 'CreateFunction20150331':
            lambda_client = boto3.client('lambda')
            arn = f"arn:aws:lambda:{region}:{account_id}:function:{resource_arn}"
            lambda_client.tag_resource(Resource=arn, Tags=tags)

        # Kafka
        elif source == 'aws.kafka' and event_name == 'CreateCluster':
            kafka_client = boto3.client("kafka")
            kafka_client.tag_resource(ResourceArn=resource_arn,Tags=tags)

        # ServiceCatalog
        elif source == 'aws.servicecatalog' and event_name == 'CreatePortfolio':
            client = boto3.client('servicecatalog')
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            client.update_portfolio(Id=resource_arn,AddTags=tag_list)

        elif source == 'aws.servicecatalog' and event_name == 'CreateProduct':
            client = boto3.client('servicecatalog')
            response = client.describe_product_as_admin(Name=resource_arn)
            product_id = response.get("ProductViewDetail", {}).get("ProductViewSummary", {}).get("ProductId", {})
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            client.update_product(Id=product_id,AddTags=tag_list)

        # DMS
        elif source == 'aws.dms' and event_name == 'CreateReplicationInstance':
            dms = boto3.client("dms")
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            dms.add_tags_to_resource(ResourceArn=resource_arn,Tags=tag_list)
         
        elif source == 'aws.dms' and event_name == 'CreateEndpoint':
            dms = boto3.client("dms")
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            dms.add_tags_to_resource(ResourceArn=resource_arn,Tags=tag_list)

        elif source == 'aws.dms' and event_name == 'CreateReplicationTask':
            dms = boto3.client("dms")
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            dms.add_tags_to_resource(ResourceArn=resource_arn,Tags=tag_list)

        # Codepipeline
        elif source == 'aws.codepipeline' and event_name == 'CreatePipeline':
            codepipeline_client = boto3.client('codepipeline')
            pipliene_arn = f'arn:aws:codepipeline:{region}:{account_id}:{resource_arn}'
            tag_list = [{'key': k, 'value': v} for k, v in tags.items()]
            codepipeline_client.tag_resource(resourceArn=pipliene_arn,tags=tag_list)

        # IAM
        elif source == 'aws.iam' and event_name == 'CreateRole':
            iam_client = boto3.client('iam')
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            response = iam_client.tag_role(RoleName=resource_arn,Tags=tag_list)

        elif source == 'aws.iam' and event_name == 'CreatePolicy':
            iam_client = boto3.client('iam')
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            response = iam_client.tag_policy(PolicyArn=resource_arn,Tags=tag_list)

        # EventBridge
        elif source == 'aws.events' and event_name == 'CreateEventBus':
            event_client = boto3.client('events')
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            event_client.tag_resource(ResourceARN=resource_arn,Tags=tag_list)

        elif source == 'aws.events' and event_name == 'PutRule':
            event_client = boto3.client('events')
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            event_client.tag_resource(ResourceARN=resource_arn,Tags=tag_list)

        # VPC
        elif service == 'ec2' and event_name == 'CreateVpc':
            ec2 = boto3.client('ec2')
            resource_id = resource_arn
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            ec2.create_tags(Resources=[resource_id], Tags=tag_list)

        # Subnet
        elif service == 'ec2' and event_name == 'CreateSubnet':
            ec2 = boto3.client('ec2')
            resource_id = resource_arn
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            ec2.create_tags(Resources=[resource_id], Tags=tag_list)
   
        print(f"Added tags to {resource_arn}")
    except Exception as e:
        print(f"Error adding tags to {service}: {e}")

# --------------------------------------- End ------------------------------------------------- #

