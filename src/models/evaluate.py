"""Evaluation script for measuring model accuracy."""

import json
import os
import tarfile
import logging
#import pickle

# Imports necessary libraries for data manipulation, modeling, and performance evaluation
import pandas as pd
import xgboost

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# May need to import additional metrics depending on what you are measuring.
# See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

if __name__ == "__main__":
    
    #Set all script variables
    model_path = "/opt/ml/processing/model/model.tar.gz"
    model_base = "/opt/ml/processing/model"
    model_fn = f"{model_base}/xgboost-model"
    
    # Extracts machine learning model from a tar file and saves it to the specified directory
    with tarfile.open(model_path) as tar:
        tar.extractall(path=model_base)

    logger.debug("Loading xgboost model.")
    ## WARNING: there is a bug with XGBClassifier on this docker image
    # The settings from the original model file are not being picked up, you need to explicitly set the objective and num_class again
    # Loads a XGBoost classification model from the specified file path.
    model = xgboost.XGBClassifier(objective="multi:softmax", num_class=5)
    model.load_model(model_fn)

    # Read test dataset into two separate dataframes, 1 for inferente test and 1 for comparative result test
    logger.debug("Loading testing data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    dfTest = pd.read_csv(test_path, header=None)

    # Extract the target variable and feature matrix from the test dataframe.
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = df.values

    # Run predictions
    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)
    predict_proba = model.predict_proba(X_test)

    # Log sample of predictions vs real
    logger.info("Sample predictions:")
    logger.info(predictions[:50])
    logger.info("Sample y test:")
    logger.info(y_test[:50])

    # Creating classification evaluation report
    print("Creating classification evaluation report")
    acc = accuracy_score(y_test, predictions.round())

    # The metrics reported can change based on the model used, but it must be a specific name per (https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html)
    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {
                "value": acc,
                "standard_deviation" : "NaN"
            },
        },
    }

    # Writes the evaluation report as JSON to a specified file path
    evaluation_output_path = '/opt/ml/processing/evaluation/evaluation.json'
    with open(evaluation_output_path, 'w') as f:
        f.write(json.dumps(report_dict))