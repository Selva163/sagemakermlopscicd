import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_transformer

from sklearn.exceptions import DataConversionWarning

from io import StringIO # python3; python2: BytesIO 
import boto3

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


columns = [
    "age",
    "education",
    "major industry code",
    "class of worker",
    "num persons worked for employer",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "income",
]
class_labels = [" - 50000.", " 50000+."]


def write_dataset_to_path(tt_features,tt_labels,features_output_path,labels_output_path,monitor_infer_path,feature_columns):
    print("Saving features to {}".format(features_output_path))
    adf = pd.DataFrame(tt_features,columns=feature_columns)
    adf.to_csv(monitor_infer_path,header=False, index=False)
    adf['income'] = tt_labels
    adf.to_csv(features_output_path, index=False)

    print("Saving labels to {}".format(labels_output_path))
    tt_labels.to_csv(labels_output_path, header=False, index=False)

def write_artifacts_local_mode(bucket,obj_to_write, filename_to_write):
    csv_buffer = StringIO()
    obj_to_write.to_csv(csv_buffer, header=False, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, filename_to_write).put(Body=csv_buffer.getvalue())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    parser.add_argument('--runtype', type=str, default="notest")
    parser.add_argument('--testbucket', type=str, default="")

    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    input_data_path = os.path.join("/opt/ml/processing/input", "census-income.csv")

    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path)
    df = pd.DataFrame(data=df, columns=columns)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.replace(class_labels, [0, 1], inplace=True)

    negative_examples, positive_examples = np.bincount(df["income"])
    print(
        "Data after cleaning: {}, {} positive examples, {} negative examples".format(
            df.shape, positive_examples, negative_examples
        )
    )

    split_ratio = args.train_test_split_ratio
    print("Splitting data into train and test sets with ratio {}".format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("income", axis=1), df["income"], test_size=split_ratio, random_state=0
    )

    preprocess = make_column_transformer(
        ( KBinsDiscretizer(encode="onehot-dense", n_bins=10),
            ["age", "num persons worked for employer"]
        ),
        ( StandardScaler(), ["capital gains", "capital losses", "dividends from stocks"]),
        (OneHotEncoder(sparse=False), ["education", "major industry code", "class of worker"]),
    )
    print("Running preprocessing and feature engineering transformations")
    train_features = preprocess.fit_transform(X_train)
    test_features = preprocess.transform(X_test)

    print("Train data shape after preprocessing: {}".format(train_features.shape))
    print("Test data shape after preprocessing: {}".format(test_features.shape))
    
    #paths for training and test dataset and labels.
    train_features_output_path = os.path.join("/opt/ml/processing/train", "train_features.csv")
    train_labels_output_path = os.path.join("/opt/ml/processing/train", "train_labels.csv")
    test_features_output_path = os.path.join("/opt/ml/processing/test", "test_features.csv")
    test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")
    #writing only the features for monitoring and inference purpose
    baseline_output_path = os.path.join("/opt/ml/processing/monitor", "train_features.csv")
    infer_output_path = os.path.join("/opt/ml/processing/infer", "test_features.csv")

    write_dataset_to_path(train_features,y_train,train_features_output_path,train_labels_output_path,baseline_output_path,preprocess.get_feature_names_out())
    write_dataset_to_path(test_features,y_test,test_features_output_path,test_labels_output_path,infer_output_path,preprocess.get_feature_names_out())
    
    if args.runtype == "test":
        bucket = args.testbucket # already created on S3
        write_artifacts_local_mode(bucket,pd.DataFrame(train_features), 'train_features.csv')
        write_artifacts_local_mode(bucket,y_train, 'train_labels.csv')
        write_artifacts_local_mode(bucket,pd.DataFrame(test_features), 'test_features.csv')
        write_artifacts_local_mode(bucket,y_test, 'test_labels.csv')