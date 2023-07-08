from sagemaker.workflow.parameters import ParameterString
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.model import Model
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.transformer import Transformer
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.inputs import TransformInput
from sagemaker.workflow.steps import TransformStep
import sagemaker
import boto3
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker import get_execution_role
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
import os
from sagemaker.model_monitor.dataset_format import DatasetFormat

sm_client = boto3.client(service_name="sagemaker")
region = os.environ['AWS_DEFAULT_REGION']
role = os.environ['AWS_SAGEMAKER_ROLE']
testbucket = os.environ['AWS_TEST_BUCKET']

sagemaker_session = sagemaker.Session()
default_bucket = sagemaker_session.default_bucket()
batch_data_uri = "s3://"
batch_data = ParameterString(
    name="BatchData",
    default_value=batch_data_uri,
)
model_package_group_name = 'sklearn-check-model-reg'
lambda_function_name = "get-latest-version"
pipeline_name = "inference-pipeline"

def get_pipeline_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )
pipeline_session = get_pipeline_session(region, default_bucket)

# Lambda helper class can be used to create the Lambda function
func = Lambda(
    function_name=lambda_function_name,
    execution_role_arn=role,
    script="scripts/lambda_step_code.py",
    handler="lambda_step_code.handler",
    timeout=600,
    memory_size=128,
)

step_latest_model_fetch = LambdaStep(
    name="fetchLatestModel",
    lambda_func=func,
    inputs={
        "model_package_group_name": model_package_group_name,
    },
    outputs=[
        LambdaOutput(output_name="ModelUrl", output_type=LambdaOutputTypeEnum.String), 
        LambdaOutput(output_name="ImageUri", output_type=LambdaOutputTypeEnum.String), 
        LambdaOutput(output_name="BaselineStatisticsS3Uri", output_type=LambdaOutputTypeEnum.String), 
        LambdaOutput(output_name="BaselineConstraintsS3Uri", output_type=LambdaOutputTypeEnum.String), 
    ],
)


sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1", role=role, instance_type="ml.t3.medium", instance_count=1
)

step_process = ProcessingStep(
    name="LoadInferenceData",
    code="scripts/process.py",
    processor=sklearn_processor,
    outputs=[
        ProcessingOutput(output_name="inference", source="/opt/ml/processing/inference")
    ],
    job_arguments = ['--train-test-split-ratio', '0.2', 
    '--testbucket', testbucket
    ]
)

step_infer = ProcessingStep(
    name="Inference",
    code="scripts/score.py",
    processor=sklearn_processor,
    inputs=[
            ProcessingInput(
                source=step_latest_model_fetch.properties.Outputs["ModelUrl"],
                destination="/opt/ml/processing/model",
                ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["inference"].S3Output.S3Uri, 
                destination="/opt/ml/processing/test"
                )
    ],
    job_arguments = ['--testbucket', testbucket
    ]
)

pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        batch_data,
    ],
    steps=[step_latest_model_fetch,step_process, step_infer],
)

import json
definition = json.loads(pipeline.definition())
pipeline.upsert(role_arn=role)