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
pipeline_name = "test102-batch"

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

check_job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        volume_size_in_gb=120,
        sagemaker_session=pipeline_session,
        env = {
                "PipelineName": pipeline_name,
                "Region": region,
            }
    )

data_quality_check_config = DataQualityCheckConfig(
        baseline_dataset=f's3://{testbucket}/test_features.csv',
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=f"s3://{testbucket}/models_baselines_results/",
        post_analytics_processor_script='scripts/postprocess_monitor_script.py',
    )

data_quality_check_step = QualityCheckStep(
    name="DataQualityCheckStep",
    skip_check=False,
    register_new_baseline=False,
    quality_check_config=data_quality_check_config,
    check_job_config=check_job_config,
    supplied_baseline_statistics=step_latest_model_fetch.properties.Outputs["BaselineStatisticsS3Uri"],
    supplied_baseline_constraints=step_latest_model_fetch.properties.Outputs["BaselineConstraintsS3Uri"],
    model_package_group_name=model_package_group_name
)

sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1", role=role, instance_type="ml.m5.xlarge", instance_count=1
)

step_infer = ProcessingStep(
    name="Inference",
    code="scripts/score.py",
    processor=sklearn_processor,
    inputs=[
            ProcessingInput(
                source=step_latest_model_fetch.properties.Outputs["ModelUrl"],
                destination="/opt/ml/processing/model",
            )
    ],
    job_arguments = ['--testbucket', testbucket
    ]
)

# model = Model( 
#     image_uri=step_latest_model_fetch.properties.Outputs["ImageUri"],
#     model_data=step_latest_model_fetch.properties.Outputs["ModelUrl"],
#     sagemaker_session=sagemaker_session,
#     role=role,
# )

# inputs = CreateModelInput(
#     instance_type="ml.m5.large",
# )
# step_create_model = CreateModelStep(
#     name="CreateModel",
#     model=model,
#     inputs=inputs,
# )

# transformer = Transformer(
#     model_name=step_create_model.properties.ModelName,
#     instance_type="ml.m5.large",
#     instance_count=1,
#     output_path=f"s3://{default_bucket}/AbaloneTransform",
# )

# step_transform = TransformStep(
#     name="Transform", transformer=transformer, inputs=TransformInput(data=batch_data)
# )

pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        batch_data,
    ],
    steps=[step_latest_model_fetch, step_infer],
)

import json
definition = json.loads(pipeline.definition())
pipeline.upsert(role_arn=role)