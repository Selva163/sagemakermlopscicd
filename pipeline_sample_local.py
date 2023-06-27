import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.processing import SKLearnProcessor
import json
from sagemaker.s3 import S3Downloader
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
import os 
from sagemaker.workflow.pipeline_context import LocalPipelineSession

local_pipeline_session = LocalPipelineSession()
region = os.environ['AWS_DEFAULT_REGION']
role = os.environ['IAM_ROLE_NAME']
testbucket = os.environ['AWS_TEST_BUCKET']
plname = "test102"
processing_instance = "ml.m5.xlarge"
input_data = "s3://sagemaker-sample-data-{}/processing/census/census-income.csv".format(region)


sklearn_processor = SKLearnProcessor(
    framework_version="0.20.0", 
    role=role,
    instance_type=processing_instance, 
    instance_count=1, 
    sagemaker_session = local_pipeline_session
)


step_process = ProcessingStep(
    name="Process",
    code="scripts/process.py",
    processor=sklearn_processor,
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test"),
    ],
    job_arguments = ['--train-test-split-ratio', '0.2', 
    '--runtype', 'test',
    '--testbucket', testbucket
    ]
)

sklearn = SKLearn(
    entry_point="scripts/train.py", 
    framework_version="0.23-1", 
    instance_type=processing_instance, 
    role=role, 
    base_job_name="training",
    hyperparameters = {'solver': 'lbfgs', 'runtype': 'test', 'testbucket': testbucket}
)

step_train = TrainingStep(
    name="TrainStep",
    estimator=sklearn
)
step_train.add_depends_on([step_process])

step_evaluate = ProcessingStep(
    name="Evaluate",
    code="scripts/evaluate.py",
    processor=sklearn_processor,
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test"),
    ],
    job_arguments = ['--runtype', 'test',
    '--testbucket', testbucket
    ]
)
step_evaluate.add_depends_on([step_train])


pipeline = Pipeline(
    name = plname,
    steps=[step_process,step_train,step_evaluate],
    sagemaker_session=local_pipeline_session
)

pipeline.upsert(role_arn=role)
# execution=pipeline.start()