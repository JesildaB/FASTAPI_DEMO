import os
import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError


def get_runtime(region_name=None):
    region = region_name or os.getenv("AWS_REGION")
    return boto3.client("sagemaker-runtime", region_name=region)


def invoke_endpoint(runtime, endpoint_name, payload, content_type="text/csv"):
    if isinstance(payload, (list, dict)):
        # convert list of floats into CSV row
        body = ",".join([str(x) for x in payload])
    else:
        body = str(payload)

    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Body=body
        )
        result = response["Body"].read().decode("utf-8")
        return result
    except (BotoCoreError, ClientError) as e:
        raise
