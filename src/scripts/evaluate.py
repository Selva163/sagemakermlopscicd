import json
import os
import sys 
os.system(f"{sys.executable} -m pip install s3fs")
os.system(f"{sys.executable} -m pip install fsspec")
os.system(f"{sys.executable} -m pip install -U scikit-learn")
os.system(f"{sys.executable} -m pip install boto3")
os.system(f"{sys.executable} -m pip install sagemaker")
import joblib
from time import gmtime, strftime
import tarfile
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)
from io import StringIO # python3; python2: BytesIO 
import boto3
import io
import pickle
import pathlib
import argparse
from sagemaker.experiments.run import Run,load_run
from sagemaker.session import Session

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--runtype', type=str, default="notest")
    parser.add_argument('--testbucket', type=str, default="")
    parser.add_argument('--experiment-name', type=str, default="")
    parser.add_argument('--run-name', type=str, default="")
    parser.add_argument('--region', type=str, default="")

    args, _ = parser.parse_known_args()

    s3 = boto3.resource('s3')
    
    model_path = os.path.join("/opt/ml/processing/model", "model.tar.gz")
    print("Extracting model from path: {}".format(model_path))
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    print("Loading model")
    model = joblib.load("model.joblib")

    print("Loading test input data")
    
    test_features_data = os.path.join("/opt/ml/processing/test", "test_features.csv")
    test_labels_data = os.path.join("/opt/ml/processing/test", "test_labels.csv")

    X_test = pd.read_csv(test_features_data).drop("income", axis=1)
    y_test = pd.read_csv(test_labels_data, header=None)

    predictions = model.predict(X_test)
    prediction_probabilities = model.predict_proba(X_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    lr_probs = prediction_probabilities[:, 1]
    fpr, tpr, _ = roc_curve(y_test, lr_probs)
    roc_auc = roc_auc_score(y_test, predictions)
    
    # Available metrics to add to model: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    print("Creating classification evaluation report")
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "confusion_matrix": {
                "0": {"0": int(conf_matrix[0][0]), "1": int(conf_matrix[0][1])},
                "1": {"0": int(conf_matrix[1][0]), "1": int(conf_matrix[1][1])},
            },
            "receiver_operating_characteristic_curve": {
                "false_positive_rates": list(fpr),
                "true_positive_rates": list(tpr),
            },
            "roc_auc": {
                "value": roc_auc, "standard_deviation": "NaN"
            }
        },
    }

    print("Classification report:\n{}".format(report_dict))
    
    csv_buffer3 = StringIO()
    pd.DataFrame(predictions).to_csv(csv_buffer3, header=False, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(args.testbucket, 'test_pred.csv').put(Body=csv_buffer3.getvalue())

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
    
    if args.runtype == "test":
        s3 = boto3.resource('s3')
        s3object = s3.Object(args.testbucket, 'report_dict.json')

        s3object.put(
        Body=(bytes(json.dumps(report_dict).encode('UTF-8')))
        )
    else:
        preds_output_path = os.path.join("/opt/ml/processing/prediction", "predictions.csv")
        X_test["income"] = y_test
        X_test["income_pred"] = predictions
        X_test.to_csv(preds_output_path, index=False)
        
        boto_session = boto3.session.Session(region_name=args.region)
        sagemaker_session = Session(boto_session=boto_session)
        dtimem = gmtime()
        fg_ts_str = str(strftime("%Y%m%d%H%M%S", dtimem))
        run_name = "evaluate-"+fg_ts_str
        with Run(experiment_name=args.experiment_name, run_name=run_name, sagemaker_session=sagemaker_session) as run:
            run.log_parameters(
                {"C": model.C, "solver": model.solver, "penalty": model.penalty, "runtype": 'evaluate', 'device':'cpu'}
                )
            run.log_metric("accuracy", accuracy)