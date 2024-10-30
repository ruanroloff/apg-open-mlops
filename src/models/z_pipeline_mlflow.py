''' not working
import os
import datetime

from models._utils import get_session

import sagemaker
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.function_step import step
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="apg-model-pkg-heart-mflow",  # Não usado
    pipeline_name="apg-pipe-heart-p-mflow",  # You can find your pipeline name in the Studio UI (project -> Pipelines -> name)
    base_job_prefix="apg-job-heart-mflow",  # Choose any name
    inputdata=None,
    modelpath=None,
    mlflowsrv=None
):

    datenow = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")

    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    
    print("role sm: " + role)
    bucket = sagemaker_session.default_bucket()
    #region = sagemaker_session.boto_region_name

    #pipeline_name = "breast-cancer-xgb"
    instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )

    # Mlflow (replace these values with your own)
    #tracking_server_arn = "your tracking server arn"
    #experiment_name = "sm-pipelines-experiment"
    #tracking_server_arn = f"arn:aws:sagemaker:us-east-1:828238096174:mlflow-tracking-server/apg-mlflow-sage"
    tracking_server_arn = mlflowsrv
    experiment_name = "exper-heart-mlflow"
    model_threshold = 0.6

    # Location of our dataset
    #input_path = f"s3://sagemaker-example-files-prod-{region}/datasets/tabular/breast_cancer/wdbc.csv"
    #ds_name = 'full_heart_disease_uci.csv'
    #input_path = f"s3://{bucket}/datasets/heart/{ds_name}"
    input_path = inputdata

    from sagemaker import image_uris
    #image_uri = image_uris.retrieve(framework='sklearn',region='us-east-1',version='0.23-1',image_scope='inference')
    image_uri = image_uris.retrieve(framework='sklearn',region='us-east-1',version='1.2-1',image_scope='training')

    #internal module features.py
    from models.features import target_variable, feature_names
    label_column = target_variable
    feature_names = feature_names

    print("START STEP PROCESS")
    print("\n")
    ## step preprocess
    @step(
        name="APG-MFLOW-DataPreprocessing",
        image_uri=image_uri,
        instance_type=instance_type,
        role=role,
    )
    def preprocess(
        raw_data_s3_path: str,
        output_prefix: str,
        experiment_name: str,
        run_name: str,
        test_size: float = 0.2,
    ) -> tuple:
        import mlflow
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler


        mlflow.set_tracking_uri(tracking_server_arn)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            with mlflow.start_run(run_name="DataPreprocessing", nested=True):
                df = pd.read_csv(raw_data_s3_path)
                #df = pd.read_csv(raw_data_s3_path, header=None, names=feature_names)
                #df.drop(columns="id", inplace=True)
                mlflow.log_input(
                    mlflow.data.from_pandas(df, raw_data_s3_path, targets=label_column),
                    context="DataPreprocessing",
                )

                X = df.drop(label_column, axis=1)
                y = df[label_column]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                #train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[label_column])
                #validation_df, test_df = train_test_split(
                #    test_df, test_size=0.5, stratify=test_df[label_column]
                #)
                #train_df.reset_index(inplace=True, drop=True)
                #validation_df.reset_index(inplace=True, drop=True)
                #test_df.reset_index(inplace=True, drop=True)

                X_train = StandardScaler().fit_transform(X_train)
                X_test = StandardScaler().transform(X_test)

                Xtrain_s3_path = f"s3://{bucket}/{output_prefix}/xtrain.csv"
                Ytrain_s3_path = f"s3://{bucket}/{output_prefix}/ytrain.csv"

                Xtest_s3_path = f"s3://{bucket}/{output_prefix}/xtrain.csv"
                Ytest_s3_path = f"s3://{bucket}/{output_prefix}/ytrain.csv"

                #train_s3_path = f"s3://{bucket}/{output_prefix}/train.csv"
                #y_train_s3_path = f"s3://{bucket}/{output_prefix}/train-y.csv"
                #val_s3_path = f"s3://{bucket}/{output_prefix}/validation.csv"
                #test_s3_path = f"s3://{bucket}/{output_prefix}/test.csv"

                #X_train.to_csv(train_s3_path, index=False)
                #y_train.to_csv(y_train_s3_path, index=False)
                #y_test.to_csv(val_s3_path, index=False)
                #X_test.to_csv(test_s3_path, index=False)

        return Xtrain_s3_path, Ytrain_s3_path, Xtest_s3_path, Ytest_s3_path, experiment_name, run_id

    ## param step train
    use_gpu = False
    param = dict(
        multi_class="auto", 
        C=1, 
        penalty="l2", 
        max_iter=1000
    )
    #num_round = 50

    ## step training
    @step(
        name="APG-MFLOW-ModelTraining",
        image_uri=image_uri,
        instance_type=instance_type,
        role=role,
    )
    def train(
        xtrain_s3_path: str,
        ytrain_s3_path: str,
        xtest_s3_path: str,
        ytest_s3_path: str,
        experiment_name: str,
        run_id: str,
        param: dict = param,
    ):
        import mlflow
        import pandas as pd
        #from xgboost import XGBClassifier
        from sklearn.linear_model import LogisticRegression


        mlflow.set_tracking_uri(tracking_server_arn)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_id=run_id):
            with mlflow.start_run(run_name="ModelTraining", nested=True) as training_run:
                training_run_id = training_run.info.run_id
                mlflow.sklearn.autolog(
                    log_input_examples=True,
                    log_model_signatures=True,
                    log_models=True,
                    log_datasets=True,
                )

                # read data files from S3
                X_train = pd.read_csv(xtrain_s3_path)
                X_test = pd.read_csv(xtest_s3_path)
                #train_df = pd.read_csv(train_s3_path)
                #validation_df = pd.read_csv(test_s3_path)

                # create dataframe and label series
                y_train = pd.read_csv(ytrain_s3_path)
                y_test= pd.read_csv(ytest_s3_path)
                #y_train = (train_df.pop(label_column) == "M").astype("int")
                #y_validation = (validation_df.pop(label_column) == "M").astype("int")
                

                lr = LogisticRegression(**param)
                lr.fit(
                    X_train, 
                    y_train,
                    eval_set=[(X_test, y_test)],)
                #xgb = XGBClassifier(n_estimators=num_round, **param)
                #xgb.fit(
                #    train_df,
                #    y_train,
                #    eval_set=[(validation_df, y_validation)],
                #    early_stopping_rounds=5,
                #)

            # return xgb
            return experiment_name, run_id, training_run_id
        
    ## step model eval
    @step(
        name="APG-MFLOW-ModelEvaluation",
        image_uri=image_uri,
        instance_type=instance_type,
        role=role,
    )
        
    def evaluate(
        xtest_s3_path: str,
        ytest_s3_path: str,
        experiment_name: str,
        run_id: str,
        training_run_id: str,
    ) -> dict:
        import mlflow
        import pandas as pd

        from sklearn.metrics import accuracy_score

        mlflow.set_tracking_uri(tracking_server_arn)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_id=run_id):
            with mlflow.start_run(run_name="ModelEvaluation", nested=True):
                #test_df = pd.read_csv(test_s3_path)
                #test_df[label_column] = (test_df[label_column] == "M").astype("int")

                X_test = pd.read_csv(xtest_s3_path)
                y_test = pd.read_csv(ytest_s3_path)
                model = mlflow.pyfunc.load_model(f"runs:/{training_run_id}/model")

                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                #results = mlflow.evaluate(
                #    model=model,
                #    data=test_df,
                #    targets=label_column,
                #    model_type="classifier",
                #    evaluators=["default"],
                #)
                #return {"accuracy": results.metrics["accuracy"]}
                return {"accuracy": accuracy}

    ## step model register    
    @step(
        name="APG-MFLOW-ModelRegistration",
        image_uri=image_uri,
        instance_type=instance_type,
        role=role,
    )
    def register(
        pipeline_name: str,
        experiment_name: str,
        run_id: str,
        training_run_id: str,
    ):
        import mlflow

        mlflow.set_tracking_uri(tracking_server_arn)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_id=run_id):
            with mlflow.start_run(run_name="ModelRegistration", nested=True):
                mlflow.register_model(f"runs:/{training_run_id}/model", pipeline_name)

    ## attach pipeline steps
    preprocessing_step = preprocess(
        raw_data_s3_path=input_path,
        output_prefix=f"{pipeline_name}/dataset",
        experiment_name=experiment_name,
        run_name=ExecutionVariables.PIPELINE_EXECUTION_ID,
    )

    training_step = train(
        #train_s3_path=preprocessing_step[0],
        #validation_s3_path=preprocessing_step[1],
        xtrain_s3_path=preprocessing_step[0],
        ytrain_s3_path=preprocessing_step[1],
        xtest_s3_path=preprocessing_step[2],
        ytest_s3_path=preprocessing_step[3],
        experiment_name=preprocessing_step[4],
        run_id=preprocessing_step[6],
    )

    conditional_register_step = ConditionStep(
        name="APG-MFLOW-ConditionalRegister",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=evaluate(
                    #test_s3_path=preprocessing_step[2],
                    xtest_s3_path=preprocessing_step[2],
                    ytest_s3_path=preprocessing_step[3],
                    experiment_name=preprocessing_step[4],
                    run_id=preprocessing_step[4],
                    training_run_id=training_step[2],
                )["accuracy"],
                #right=0.7,
                right=model_threshold,
            )
        ],
        if_steps=[
            register(
                pipeline_name=pipeline_name,
                experiment_name=preprocessing_step[3],
                run_id=preprocessing_step[4],
                training_run_id=training_step[2],
            )
        ],
        else_steps=[FailStep(name="Fail", error_message="Model performance is not good enough")],
    )

    ## create pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            instance_type,
        ],
        steps=[preprocessing_step, training_step, conditional_register_step],
        sagemaker_session=sagemaker_session,
    )

    pipeline.upsert(role_arn=role)

    return pipeline
'''