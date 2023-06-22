import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import pickle

if __name__ == "__main__":
    training_data_directory = "/opt/ml/processing/train"
    test_data_directory = "/opt/ml/processing/test"
    train_features_data = os.path.join(training_data_directory, "train_features.csv")
    train_labels_data = os.path.join(training_data_directory, "train_labels.csv")
    print("Reading input data")
    X_train = pd.read_csv(train_features_data, header=None)
    y_train = pd.read_csv(train_labels_data, header=None)

    model = LogisticRegression(class_weight="balanced", solver="lbfgs")
    print("Training LR model")
    model.fit(X_train, y_train)
    #model_output_directory = os.path.join("/opt/ml/model", "model.joblib")
    #print("Saving model to {}".format(model_output_directory))
    #joblib.dump(model, model_output_directory)
    import boto3
    s3_resource = boto3.resource('s3')
    bucket='s3tmc101'
    key= 'pickle_model.pkl'

    pickle_byte_obj = pickle.dumps(model)

    s3_resource.Object(bucket,key).put(Body=pickle_byte_obj)
