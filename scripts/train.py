import os
import sys 
os.system(f"{sys.executable} -m pip install s3fs")
os.system(f"{sys.executable} -m pip install fsspec")
os.system(f"{sys.executable} -m pip install -U scikit-learn")
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import pickle
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--solver', type=str, default="lbfgs")
    parser.add_argument('--runtype', type=str, default="notest")
    parser.add_argument('--testbucket', type=str, default="")

    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args, _ = parser.parse_known_args()

    training_data_directory = "/opt/ml/processing/train"
    test_data_directory = "/opt/ml/processing/test"
    train_features_data = os.path.join(training_data_directory, "train_features.csv")
    train_labels_data = os.path.join(training_data_directory, "train_labels.csv")
    print("Reading input data")
    X_train = pd.read_csv(f's3://{args.testbucket}/train_features.csv', header=None)
    y_train = pd.read_csv(f's3://{args.testbucket}/train_labels.csv', header=None)

    model = LogisticRegression(class_weight="balanced", solver=args.solver)
    print("Training LR model")
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    if args.runtype == "test":
        import boto3
        s3_resource = boto3.resource('s3')
        bucket=args.testbucket
        key= 'pickle_model.pkl'

        pickle_byte_obj = pickle.dumps(model)

        s3_resource.Object(bucket,key).put(Body=pickle_byte_obj)
