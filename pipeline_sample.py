#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
from sagemaker.workflow.step_collections import RegisterModel
import os 
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


region = 'us-east-1' #os.environ['AWS_DEFAULT_REGION']

role = 'arn:aws:iam::625594729569:role/service-role/AmazonSageMaker-ExecutionRole-20230222T105014' #os.environ['IAM_ROLE_NAME']
sklearn_processor = SKLearnProcessor(
    framework_version="0.20.0", role=role, instance_type="ml.t3.medium", instance_count=1
)


# In[5]:


input_data = "s3://sagemaker-sample-data-{}/processing/census/census-income.csv".format(region)


# In[6]:


step_process = ProcessingStep(
    name="Process",
    code="process.py",
    processor=sklearn_processor,
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test"),
    ]
)


# In[7]:


sklearn = SKLearn(
    entry_point="train.py", framework_version="0.23-1", instance_type="ml.m4.xlarge", role=role, base_job_name="training"
)

step_train = TrainingStep(
    name="TrainStep",
    estimator=sklearn
)
step_train.add_depends_on([step_process])
# sklearn.fit({"train": preprocessed_training_data})
# training_job_description = sklearn.jobs[-1].describe()
# model_data_s3_uri = "{}{}/{}".format(
#     training_job_description["OutputDataConfig"]["S3OutputPath"],
#     training_job_description["TrainingJobName"],
#     "output/model.tar.gz",
# )


# In[8]:




evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)
step_evaluate = ProcessingStep(
    name="Evaluate",
    code="evaluate.py",
    processor=sklearn_processor,
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=Join(
                on="/",
                values=[
                    "s3://{}".format('s3tmc101'),
                    'modelprefix',
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                    "evaluation-report",
                ],
            ),
        ),
    ],
    property_files=[evaluation_report]
)
step_evaluate.add_depends_on([step_train])


model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(
            on="/",
            values=[
                step_evaluate.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                    "S3Uri"
                ],
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
    model_package_group_name="sklearn-check-model-reg",
    model_metrics=model_metrics
)

cond_gte = ConditionGreaterThanOrEqualTo(  # You can change the condition here
        left=JsonGet(
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path="roc_auc",  # This should follow the structure of your report_dict defined in the evaluate.py file.
        ),
        right=0.7,  # You can change the threshold here
)

step_cond = ConditionStep(
    name="ROCCondCheck",
    conditions=[cond_gte],
    if_steps=[step_register],
    else_steps=[]
)



# In[9]:


plname = "test102"
pipeline = Pipeline(
    name = plname,
    steps=[step_process,step_train,step_evaluate,step_cond]
)
pipeline.upsert(role_arn=role)
execution=pipeline.start()


# In[ ]:


# sklearn_processor.run(
#     code="evaluation.py",
#     inputs=[
#         ProcessingInput(source=model_data_s3_uri, destination="/opt/ml/processing/model"),
#         ProcessingInput(source=preprocessed_test_data, destination="/opt/ml/processing/test"),
#     ],
#     outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
# )

