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
from sklearn.model_selection import GridSearchCV
from time import gmtime, strftime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--solver', type=str, default="lbfgs")
    parser.add_argument('--runtype', type=str, default="notest")
    parser.add_argument('--testbucket', type=str, default="")
    parser.add_argument('--experiment-name', type=str, default="")
    parser.add_argument('--region', type=str, default="")

    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args, _ = parser.parse_known_args()
    

    training_data_directory = "/opt/ml/input/data/train"
    train_features_data = os.path.join(training_data_directory, "train_features.csv")
    train_labels_data = os.path.join(training_data_directory, "train_labels.csv")
    print("Reading input data")
    X_train = pd.read_csv(train_features_data).drop("income", axis=1)
    y_train = pd.read_csv(train_labels_data, header=None)

    logreg=LogisticRegression(solver = 'liblinear')
    parameters = [{'solver': ['lbfgs', 'liblinear'],
                'penalty':[ 'elasticnet', 'l1', 'l2'],
                'C':[ 0.01, 0.1, 1, 10]}]
    grid_search = GridSearchCV(estimator = logreg,  
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 5,
                            verbose=0)
    grid_search.fit(X_train, y_train)

    print("Training LR model")
    model = grid_search.best_estimator_
    cv_results = grid_search.cv_results_
    df = pd.DataFrame(cv_results)
    hypertuning_results_list = df[df['mean_test_score'].notna()][['param_C', 'param_solver', 'param_penalty', 'mean_test_score']].values
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    if args.runtype == "notest":
        boto_session = boto3.session.Session(region_name=args.region)
        sagemaker_session = Session(boto_session=boto_session)
        for hr in hypertuning_results_list:
            dtimem = gmtime()
            fg_ts_str = str(strftime("%Y%m%d%H%M%S", dtimem))
            run_name = "train-"+fg_ts_str
            with Run(experiment_name=args.experiment_name, run_name=run_name, sagemaker_session=sagemaker_session) as run:
                run.log_parameters(
                    {"C": hr[0], "solver": hr[1], "penalty": hr[2], "runtype": 'train', 'device':'cpu'}
                    )
                run.log_metric("accuracy", hr[3])

    if args.runtype == "test":
        import boto3
        s3_resource = boto3.resource('s3')
        bucket=args.testbucket
        key= 'pickle_model.pkl'

        pickle_byte_obj = pickle.dumps(model)

        s3_resource.Object(bucket,key).put(Body=pickle_byte_obj)
