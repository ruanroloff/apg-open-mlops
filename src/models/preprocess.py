# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Feature engineers the heart disease dataset."""

# Import the necessary libraries and modules for the project
import argparse
import logging
import pathlib

import boto3

import numpy as np
import pandas as pd

#Import commonly used machine learning libraries for data preprocessing and model selection.
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from lightgbm import LGBMRegressor 

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

####### INSTALL PACKAGE
import subprocess
import sys
'''
This routine should not be in the code. It was added due to problems with pre-built images.
Backlog: Create custom docker with the packages already installed
'''
# Installs  Python package using the system's pip package manager.
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
####### END INSTALL PACKAGE

if __name__ == "__main__":
    ##Not a good practice
    ##
    #Install lightgbm using subprocess
    logger.info("Installing lightgbm.")
    install("lightgbm")

    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    # Set main variables for the EDA
    base_dir = "/opt/ml/processing"
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])
    fn = f"{base_dir}/data/heart_disease_uci.csv"
    logger.info("Input data: " + input_data)

    # Creates the data directory if it doesn't exist, and downloads data from S3 to the local machine.
    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    ###### Begin The EDA 
    # read in csv
    logger.info("Reading downloaded data.")
    df = pd.read_csv(fn)

    # Removes the 'id' column from the DataFrame
    df.drop('id', axis=1, inplace=True)

    # replace zero values with NaN in cholestrol column
    df['chol'] = df['chol'].replace(0, np.NaN)

    # encode the data using label encoding
    columns_to_encode = ['sex', 'dataset', 'cp', 'thal', 'slope', 'exang', 'restecg', 'fbs']
    num_cols = [col for col in df.select_dtypes(exclude='O')]
    label_encoders = {}
    data = df.copy()
    index = {}

    for colm in columns_to_encode:
        nan_ixs = np.where(data[colm].isna())[0]
        index[colm] = nan_ixs

    # Identify numeric and categorical columns with missing values in the DataFrame.
    cols_with_nans = [x for x in df if df[x].isnull().sum() > 0]
    num = [col for col in df[cols_with_nans].select_dtypes(exclude='O')]
    cat = [col for col in df[cols_with_nans].select_dtypes(include='O')]
    num

    df.ca.unique()

    #Numeric columns
    numeri_cols = ['trestbps', 'chol', 'thalch', 'oldpeak']

    logger.info("Appling Encode in categorical columns.")
    # Apply Encode categorical columns, store label encoders, and replace encoded values with NaNs.
    # The entire data set must be represented by numbers    
    for col in columns_to_encode:

        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

        for col, idxs in index.items():
            df.loc[idxs, col] = np.nan

    #List of categorical columns in the DataFrame.
    categ_cols = ['fbs', 'restecg', 'exang', 'slope', 'thal', 'ca']

    logger.info("Impute missing values in NUMeric column.")
    #Impute missing values in numeric columns using a LightGBM regression model.
    #from sklearn.preprocessing import MinMaxScaler
    for col in numeri_cols:

        df_with_missing = df[df[col].isna()]
        # dropna removes all rows with missing values
        df_without_missing = df[df[col].notna()]

        # split the data into X and y and we will only take the columns with no missing values
        X = df_without_missing.drop([col], axis=1)
        y = df_without_missing[col]

        # split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        from lightgbm import LGBMRegressor 
        # Random Forest Imputation
        rf_model = LGBMRegressor(
            objective = "regression",
            metric = "rmse",
            n_estimators =  1000,
            bagging_freq = 1,subsample = 0.413103572972995, 
                                 colsample_bytree = 0.5816717344110182,
                                 min_data_in_leaf = 20,
                                 learning_rate = 0.004730072022055302,
                                 num_leaves = 364, verbose = -1 ,random_state=42)

        rf_model.fit(X_train, y_train)

        # evaluate the model
        y_preds = rf_model.predict(X_test)

        #Apply the trained model to predict missing values
        y_pred = np.round(rf_model.predict(df_with_missing.drop([col], axis=1)))
        df_with_missing[col] = y_pred

        #concatenate the imputed DataFrame.
        df = pd.concat([df_with_missing, df_without_missing], axis=0)
    
    logger.info("Impute missing values in CATegorical column.")
    #Impute missing values in categorical columns using a LightGBM model
    for col in categ_cols: 
        df_with_missing = df[df[col].isna()]
        # dropna removes all rows with missing values
        # df_without_missing = df.dropna()
        df_without_missing = df[df[col].notna()]

        # split the data into X and y and we will only take the columns with no missing values
        X = df_without_missing.drop([col], axis=1)
        y = df_without_missing[col]

        # split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        # Random Forest Imputation
        from lightgbm import LGBMClassifier
        rf_model = LGBMClassifier(verbose = -1,learning_rate = 0.023021779601797816, num_leaves = 149, subsample = 0.6929884706542179, colsample_bytree = 0.8635308367372507, min_data_in_leaf = 47, random_state=42)
        rf_model.fit(X_train, y_train)

        # evaluate the model
        y_preds = rf_model.predict(X_test)
        y_pred = rf_model.predict(df_with_missing.drop([col], axis=1))
        acc_score = accuracy_score(y_test, y_preds)

        # print result
        print("The feature '"+ col+ "' has been imputed with", round((acc_score * 100), 2), "accuracy\n")
        df_with_missing[col] = y_pred
        
        # apply the predicted values to the missing data.
        df = pd.concat([df_with_missing, df_without_missing], axis=0)

    #df = df.fillna(0)

    logger.info("Convert categoricall to integer.")
    #Convert the encoded categorical columns to integer data type.
    for i in columns_to_encode:
        df[i] = df[i].astype(int)

    #Reverse the label encoding and restore the original categorical values in the DataFrame.
    for col in columns_to_encode:
        # Retrieve the corresponding LabelEncoder for the column
        le = label_encoders[col]
        # Inverse transform the data
        df[col] = le.inverse_transform(df[col]).astype('O')

    # it is not physiologically possible for resting blood pressure in mm Hg to be zero.
    df = df[df['trestbps'] != 0]

    # Separate numerical and categorical features
    num_cols = [col for col in df.select_dtypes(exclude='O')]
    cat_cols = [col for col in df.select_dtypes(include='O')]
    num_cols

    #Split the DataFrame into numeric and categorical columns for further processing.
    n = df[['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'num']]
    num_cols = n
    cat_cols = pd.DataFrame (df, columns= cat_cols)

    #Drop duplicate data
    df.drop_duplicates(inplace=True)   

    #Bin the numeric 'num' column into categories based on predefined thresholds.
    bins=[0,1,2,3,4]
    labels=['No-Heart-Disease', 'Mild-Heart-Disease', 'Moderate-Heart-Disease', 'Severe-Heart-Disease']
    df['num_bins']=pd.cut(df['num'], bins=bins, labels=labels, include_lowest=True)

    logger.info("Encode categorical columns using LabelEncoder.")
    #Encode categorical columns using LabelEncoder and store the encoders for later reference.
    label_encoders ={}
    for col in df[['sex', 'dataset', 'cp', 'thal', 'slope', 'exang', 'restecg', 'fbs','num_bins' ]]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    #Rearrange the columns in the DataFrame, moving the last column to the front.
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    #Drop num and keep the new bin target.
    df.drop('num', axis=1, inplace=True)
    model_data = df

    logger.info("Saving Datasets in CSV format.")
    ###SAVE datasets
    #This dataset will be used by other process
    pd.DataFrame(model_data).to_csv(
        #f"{base_dir}/modeldata/fulldataset.csv", header=True, index=False
        f"{base_dir}/processed/fulldataset.csv", header=True, index=False
    )

    # Split the data in training and test
    train_data, validation_data, test_data = np.split(
        model_data.sample(frac=1, random_state=1729),
        [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
    )

    pd.DataFrame(train_data).to_csv(
        f"{base_dir}/train/train.csv", header=False, index=False
    )
    pd.DataFrame(validation_data).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test_data).to_csv(
        f"{base_dir}/test/test.csv", header=False, index=False
    )
    
    logger.info("Uploading Datasets to S3.")
    ## Upload this other datasets to S3 bucket (not)
    ## Not necessary for the production process
    s3 = boto3.resource('s3')
    s3.Bucket(bucket).upload_file(f"{base_dir}/train/train.csv", "data/train/train.csv")
    s3.Bucket(bucket).upload_file(f"{base_dir}/validation/validation.csv", "data/validation/validation.csv")
    s3.Bucket(bucket).upload_file(f"{base_dir}/test/test.csv", "data/test/test.csv")
    s3.Bucket(bucket).upload_file(f"{base_dir}/processed/fulldataset.csv", "data/processed/fulldataset.csv")