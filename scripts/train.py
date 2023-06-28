import os
import sys 
os.system(f"{sys.executable} -m pip install s3fs")
os.system(f"{sys.executable} -m pip install fsspec")
os.system(f"{sys.executable} -m pip install -U scikit-learn")
os.system(f"{sys.executable} -m pip install boto3")
os.system(f"{sys.executable} -m pip install sagemaker")
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import pickle
import argparse
from sagemaker.experiments.run import Run,load_run
from sagemaker.session import Session
import boto3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--solver', type=str, default="lbfgs")
    parser.add_argument('--runtype', type=str, default="notest")
    parser.add_argument('--testbucket', type=str, default="")
    parser.add_argument('--experiment-name', type=str, default="")
    parser.add_argument('--run-name', type=str, default="")
    parser.add_argument('--region', type=str, default="")

    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args, _ = parser.parse_known_args()
    boto_session = boto3.session.Session(region_name=args.region)
    sagemaker_session = Session(boto_session=boto_session)
    with Run(experiment_name=args.experiment_name, run_name=args.run_name, sagemaker_session=sagemaker_session) as run:
        run.log_parameters(
            {"device": 'cpu'}
        )

    training_data_directory = "/opt/ml/input/data/train"
    train_features_data = os.path.join(training_data_directory, "train_features.csv")
    train_labels_data = os.path.join(training_data_directory, "train_labels.csv")
    print("Reading input data")
    X_train = pd.read_csv(train_features_data, header=None)
    y_train = pd.read_csv(train_labels_data, header=None)

    model = LogisticRegression(class_weight="balanced", solver=args.solver)
    print("Training LR model")
    model.fit(X_train, y_train)
    
    if args.runtype == "notest":
        joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
        with load_run(experiment_name=args.experiment_name, run_name=args.run_name, sagemaker_session=sagemaker_session) as run:
            run.log_parameters(
                {"class_weight": "balanced", "solver": args.solver, "input_size_rows": X_train.shape[0], "input_size_cols": X_train.shape[1]}
            )

    if args.runtype == "test":
        import boto3
        s3_resource = boto3.resource('s3')
        bucket=args.testbucket
        key= 'pickle_model.pkl'

        pickle_byte_obj = pickle.dumps(model)

        s3_resource.Object(bucket,key).put(Body=pickle_byte_obj)
