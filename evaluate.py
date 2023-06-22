import json
import os
import sys 
os.system(f"{sys.executable} -m pip install s3fs")
os.system(f"{sys.executable} -m pip install fsspec")
import tarfile
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from io import StringIO # python3; python2: BytesIO 
import boto3
import io
import pickle

if __name__ == "__main__":

    s3 = boto3.resource('s3')
    model = pickle.loads(s3.Bucket("s3tmc101").Object("pickle_model.pkl").get()['Body'].read())

    print("Loading test input data")
    
    X_test = pd.read_csv('s3://s3tmc101/test_features.csv', header=None)
    y_test = pd.read_csv('s3://s3tmc101/test_labels.csv', header=None)

    predictions = model.predict(X_test)

    print("Creating classification evaluation report")
    report_dict = classification_report(y_test, predictions, output_dict=True)
    report_dict["accuracy"] = accuracy_score(y_test, predictions)
    report_dict["roc_auc"] = roc_auc_score(y_test, predictions)

    print("Classification report:\n{}".format(report_dict))
    
    csv_buffer3 = StringIO()
    pd.DataFrame(predictions).to_csv(csv_buffer3, header=False, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, 'test_pred.csv').put(Body=csv_buffer3.getvalue())

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print("Saving classification report to {}".format(evaluation_output_path))
    
    s3 = boto3.resource('s3')
    s3object = s3.Object('s3tmc101', 'report_dict.json')

    s3object.put(
        Body=(bytes(json.dumps(report_dict).encode('UTF-8')))
    )