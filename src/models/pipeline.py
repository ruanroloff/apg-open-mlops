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
"""Example workflow pipeline script for Heart Diease pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""

import os
import datetime

#import boto3
import sagemaker
import sagemaker.session

#from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from models._utils import get_session
from models import train

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
datenow = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")

def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="apg-model-pkg-heart",  
    pipeline_name="apg-pipe-heart-p-ewf8t7lvhivm",
    base_job_prefix="apg-job-heart", 
    inputdata=None,
    modelpath=None,
    logpath=None,
):
    """Gets a SageMaker ML Pipeline instance working with on Heart Diease data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        model_package_group_name: 
        pipeline_name: Name of the pipeline to see in  Studio UI (domain -> Pipelines -> name)
        base_job_prefix: Part of the name of the job to see in Amazon SageMaker -> Processing jobs
        inputdata: The dataset path in S3
        modelpath: The model file output in S3
        logpath: The evaluation log bucket 
    Returns:
        an instance of a pipeline
    """    
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # Parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="Approved",  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=inputdata,  # Change this to point to the s3 location of your raw input data.
    )
    model_path = ParameterString(
        name="ModelUrl",
        default_value=modelpath
    )

    # Model evaluation path
    eval_path = f"{logpath}/eval/apg-heart-model/" + datenow
    # Defines a threshold value for model acceptance
    model_threshold = 0.6

    # Processing step for feature engineering
    feature_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-heart-preprocess", 
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    # Set parameters and attach preprocess.py  file to the step
    step_process = ProcessingStep(
        name="APG-Heart-Process", 
        processor=feature_processor,
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="processed", source="/opt/ml/processing/processed"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--input-data", input_data],
    )

    # Training step for generating model artifacts
    image_uri = train.get_image_uri(region,training_instance_type)
    xgb_train = train.get_xgb_train(region,role,sagemaker_session,training_instance_type, model_path, base_job_prefix)

    step_train = TrainingStep(
        name="APG-Heart-Train",
        estimator=xgb_train,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # Processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri = image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/eval-heart-script",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="APG-Heart-Eval",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation", destination=str(eval_path)),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )

    # Get the metrics to create condition to the Register model step 
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    # Register model step that will be conditionally executed
    step_register = RegisterModel(
        name="APG-Heart-RegisterModel",
        estimator=xgb_train,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # Condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(  
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="multiclass_classification_metrics.accuracy.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.
        ),
        # You can change the threshold at model_threshold(var)
        right=model_threshold, 
        #right=0.8,  
    )
    step_cond = ConditionStep(
        name="APG-Heart-AccuracyCond",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
            model_path,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )
    return pipeline