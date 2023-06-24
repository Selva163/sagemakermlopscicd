import json
import os
import sys 
os.system(f"{sys.executable} -m pip install s3fs")
os.system(f"{sys.executable} -m pip install fsspec")
os.system(f"{sys.executable} -m pip install -U scikit-learn")
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

if __name__ == "__main__":

    s3 = boto3.resource('s3')
    model = pickle.loads(s3.Bucket("s3tmc101").Object("pickle_model.pkl").get()['Body'].read())

    print("Loading test input data")
    
    X_test = pd.read_csv('s3://s3tmc101/test_features.csv', header=None)
    y_test = pd.read_csv('s3://s3tmc101/test_labels.csv', header=None)

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
    s3_resource.Object("s3tmc101", 'test_pred.csv').put(Body=csv_buffer3.getvalue())

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    # evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    # print("Saving classification report to {}".format(evaluation_output_path))
    
    # with open(evaluation_output_path, "w") as f:
    #     f.write(json.dumps(report_dict))
    
    #s3 = boto3.resource('s3')
    #s3object = s3.Object('s3tmc101', 'report_dict.json')

    #s3object.put(
    #    Body=(bytes(json.dumps(report_dict).encode('UTF-8')))
    #)