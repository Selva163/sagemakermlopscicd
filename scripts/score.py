"""Evaluation script for measuring mean squared error."""
import logging
import pathlib
import glob
import pickle
import tarfile
from math import sqrt
import os 
import pandas as pd
import argparse
import boto3
import joblib
import sys
from io import StringIO
os.system(f"{sys.executable} -m pip install s3fs")
os.system(f"{sys.executable} -m pip install fsspec")
os.system(f"{sys.executable} -m pip install -U scikit-learn")
logger = logging.getLogger() 
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    parser = argparse.ArgumentParser()

    parser.add_argument('--testbucket', type=str, default="")

    args, _ = parser.parse_known_args()

    s3 = boto3.resource('s3')
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading model.")
    model = joblib.load("model.joblib")

    logger.debug("Reading input data.")

    # Get input file list
    X_test = pd.read_csv(f'/opt/ml/processing/test/test_features.csv',header=None)
    predictions = model.predict(X_test)

    csv_buffer3 = StringIO()
    pd.DataFrame(predictions).to_csv(csv_buffer3, header=False, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(args.testbucket, 'predictions_batch.csv').put(Body=csv_buffer3.getvalue())
