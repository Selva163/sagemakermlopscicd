import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.processing import SKLearnProcessor
import json
from sagemaker.s3 import S3Downloader
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel
import os 
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ClarifyCheckConfig,
    ModelBiasCheckConfig,
    ModelPredictedLabelConfig,
    ModelExplainabilityCheckConfig,
    SHAPConfig
)
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig
)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep
)
from sagemaker.workflow.functions import (
    JsonGet
)
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model_metrics import MetricsSource, ModelMetrics 
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.drift_check_baselines import DriftCheckBaselines
from time import gmtime, strftime
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.steps import CreateModelStep

region = os.environ['AWS_DEFAULT_REGION']
role = os.environ['AWS_SAGEMAKER_ROLE']
testbucket = os.environ['AWS_TEST_BUCKET']
model_package_group_name = "sklearn-check-model-reg"
processing_instance = "ml.t3.medium"
training_instance = "ml.m4.xlarge"
databaseline_instance = "ml.c5.xlarge"
plname = "train-pipeline"
lambda_function_name = "get_latest_imageuri"
dtimem = gmtime()
fg_ts_str = str(strftime("%Y%m%d%H%M%S", dtimem))
experiment_name = 'sklearn-exp-101-'+fg_ts_str
base_job_prefix = "clarify"

sm_client = boto3.client('sagemaker', region_name=region)
mpg_list = [
    {"ModelPackageGroupName" : model_package_group_name,"ModelPackageGroupDescription" : "income prediction", "Tags" : [{'Key': 'team','Value': 'mlops'}]}
]

for mpg in mpg_list:
    try:
        create_mpg_response = sm_client.create_model_package_group(**mpg)
    except Exception as e:
        if 'Model Package Group already exists' in str(e):
            pass 
        else:
            raise(e)

sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1", role=role, instance_type=processing_instance, instance_count=1
)

sagemaker_session = sagemaker.Session()
featurestore_runtime_client = sagemaker_session.boto_session.client('sagemaker-runtime', region_name=region)
default_bucket = sagemaker_session.default_bucket()

def get_pipeline_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )
pipeline_session = get_pipeline_session(region, testbucket)

input_data = "s3://sagemaker-sample-data-{}/processing/census/census-income.csv".format(region)

step_process = ProcessingStep(
    name="Process",
    code="scripts/process.py",
    processor=sklearn_processor,
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test"),
        ProcessingOutput(output_name="monitor", source="/opt/ml/processing/monitor"),
        ProcessingOutput(output_name="infer", source="/opt/ml/processing/infer", destination=Join(
                on="/",
                values=[
                    "s3://{}".format(testbucket),
                    'inferencedata',
                    "income",
                ])),
    ],
    job_arguments = ['--train-test-split-ratio', '0.2', 
    '--testbucket', testbucket
    ]
)

sklearn = SKLearn(
    entry_point="scripts/train.py", 
    framework_version="1.2-1", 
    instance_type=training_instance, 
    role=role, 
    base_job_name="training",
    hyperparameters = {'solver': 'lbfgs', 'testbucket': testbucket, 'experiment-name': experiment_name , 'region': region}
)

step_train = TrainingStep(
    name="TrainStep",
    estimator=sklearn,
    inputs={
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "train_data"
            ].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

step_train.add_depends_on([step_process])

check_job_config = CheckJobConfig(
    role=role,
    instance_count=1,
    instance_type=databaseline_instance,
    volume_size_in_gb=120,
    sagemaker_session=pipeline_session,
)

data_quality_check_config = DataQualityCheckConfig(
    baseline_dataset=step_process.properties.ProcessingOutputConfig.Outputs["monitor"].S3Output.S3Uri,
    dataset_format=DatasetFormat.csv(header=False),
    output_s3_uri=Join(on='/', values=['s3:/', testbucket, 'baselinejob', ExecutionVariables.PIPELINE_EXECUTION_ID, 'dataqualitycheckstep'])
)

data_quality_check_step = QualityCheckStep(
    name="RegisterBaselineForMonitor",
    skip_check=True,
    register_new_baseline=True,
    quality_check_config=data_quality_check_config,
    check_job_config=check_job_config,
    model_package_group_name=model_package_group_name
)

drift_check_baselines = DriftCheckBaselines(
    model_data_statistics=MetricsSource(
        s3_uri=data_quality_check_step.properties.CalculatedBaselineStatistics,
        content_type="application/json",
    ),

    model_data_constraints=MetricsSource(
        s3_uri=data_quality_check_step.properties.CalculatedBaselineConstraints,
        content_type="application/json",
    )
)


evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)
step_evaluate = ProcessingStep(
    name="Evaluate",
    code="scripts/evaluate.py",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
        ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri, destination="/opt/ml/processing/test"),
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=Join(
                on="/",
                values=[
                    "s3://{}".format(testbucket),
                    'modelprefix',
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                    "evaluation-report",
                ],
            ),
        ),
        ProcessingOutput(
            output_name="predictions",
            source="/opt/ml/processing/prediction",
            destination=Join(
                on="/",
                values=[
                    "s3://{}".format(testbucket),
                    'incomemodel',
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                    "predictions",
                ],
            ),
        )
    ],
    property_files=[evaluation_report],
    job_arguments = ['--testbucket', testbucket,
    '--experiment-name', experiment_name,
    '--region', region
    ]
)
step_evaluate.add_depends_on([step_train])

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(
            on="/",
            values=[
                step_evaluate.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"],
                "evaluation.json",
            ],
        ),
        content_type="application/json",
    )
)

step_register = RegisterModel(
    name="RegisterModel",
    estimator=sklearn,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name=model_package_group_name,
    model_metrics=model_metrics,
    drift_check_baselines=drift_check_baselines,
    customer_metadata_properties={"Run":"exp-track-test","Created by":"selva"}
)

cond_gte = ConditionGreaterThanOrEqualTo(  # You can change the condition here
        left=JsonGet(
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.roc_auc.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.
        ),
        right=0.4,  # You can change the threshold here
)

step_cond = ConditionStep(
    name="MetricCheckForModelRegister",
    conditions=[cond_gte],
    if_steps=[step_register],
    else_steps=[]
)


# psteps = [step_process,step_train,data_quality_check_step,step_evaluate,step_cond,step_latest_model_fetch,step_create_model]
psteps = [step_process,step_train,data_quality_check_step,step_evaluate,step_cond]
pipeline = Pipeline(
    name = plname,
    steps=psteps
)
pipeline.upsert(role_arn=role)
# execution=pipeline.start()