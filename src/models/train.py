"""Train script for training machine learning model to the Heart Diease Use Case."""

import os
import logging

import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator

def get_image_uri(region, training_instance_type):
    image_uri = sagemaker.image_uris.retrieve(
         # we are using the Sagemaker built in xgboost algorithm
        framework="xgboost", 
        region=region,
        version="1.7-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    return image_uri

def get_xgb_train(region, role, sagemaker_session, training_instance_type, model_path, base_job_prefix):
    # Training step for generating model artifacts
    image_uri = get_image_uri(region, training_instance_type)
    
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/apg-heart-train",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    xgb_train.set_hyperparameters(
        objective="multi:softmax",
        num_class=5,
        max_depth=4,
        learning_rate=0.2,
        gamma=0.1,
        subsample=1.0,
        reg_alpha=0,
        reg_lambda=1,
        early_stopping_rounds=10,
        num_round=100,
    )
    return xgb_train

if __name__ == "__main__":
    print('xgb')