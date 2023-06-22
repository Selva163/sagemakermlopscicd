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
import os 
region = os.environ['AWS_DEFAULT_REGION']

role = os.environ['IAM_ROLE_NAME']
sklearn_processor = SKLearnProcessor(
    framework_version="0.20.0", role=role, instance_type="m4.xlarge", instance_count=1
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
    entry_point="train.py", framework_version="0.23-1", instance_type="m4.xlarge", role=role, base_job_name="training"
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


step_evaluate = ProcessingStep(
    name="Evaluate",
    code="evaluate.py",
    processor=sklearn_processor,
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test"),
    ]
)
step_evaluate.add_depends_on([step_train])


# In[9]:


plname = "test102"
pipeline = Pipeline(
    name = plname,
    steps=[step_process,step_train,step_evaluate]
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

