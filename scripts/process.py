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


def print_shape(df):
    negative_examples, positive_examples = np.bincount(df["income"])
    print(
        "Data shape: {}, {} positive examples, {} negative examples".format(
            df.shape, positive_examples, negative_examples
        )
    )


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
        (
            ["age", "num persons worked for employer"],
            KBinsDiscretizer(encode="onehot-dense", n_bins=10),
        ),
        (["capital gains", "capital losses", "dividends from stocks"], StandardScaler()),
        (["education", "major industry code", "class of worker"], OneHotEncoder(sparse=False)),
    )
    print("Running preprocessing and feature engineering transformations")
    train_features = preprocess.fit_transform(X_train)
    test_features = preprocess.transform(X_test)

    print("Train data shape after preprocessing: {}".format(train_features.shape))
    print("Test data shape after preprocessing: {}".format(test_features.shape))
    print(train_features.columns)
    print(test_features.columns)
    train_features_output_path = os.path.join("/opt/ml/processing/train", "train_features.csv")
    train_labels_output_path = os.path.join("/opt/ml/processing/train", "train_labels.csv")

    test_features_output_path = os.path.join("/opt/ml/processing/test", "test_features.csv")
    test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")

    print("Saving training features to {}".format(train_features_output_path))
    pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)

    print("Saving test features to {}".format(test_features_output_path))
    pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)

    print("Saving training labels to {}".format(train_labels_output_path))
    y_train.to_csv(train_labels_output_path, header=False, index=False)

    print("Saving test labels to {}".format(test_labels_output_path))
    y_test.to_csv(test_labels_output_path, header=False, index=False)
    
    if args.runtype == "test":
        bucket = args.testbucket # already created on S3
        csv_buffer = StringIO()
        pd.DataFrame(train_features).to_csv(csv_buffer, header=False, index=False)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket, 'train_features.csv').put(Body=csv_buffer.getvalue())
        
        csv_buffer1 = StringIO()
        y_train.to_csv(csv_buffer1, header=False, index=False)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket, 'train_labels.csv').put(Body=csv_buffer1.getvalue())

        csv_buffer2 = StringIO()
        pd.DataFrame(test_features).to_csv(csv_buffer2, header=False, index=False)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket, 'test_features.csv').put(Body=csv_buffer2.getvalue())

        csv_buffer21 = StringIO()
        pd.DataFrame(test_features).to_csv(csv_buffer21, index=False)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket, 'test_features_raw.csv').put(Body=csv_buffer21.getvalue())
        
        csv_buffer3 = StringIO()
        y_test.to_csv(csv_buffer3, header=False, index=False)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket, 'test_labels.csv').put(Body=csv_buffer3.getvalue())