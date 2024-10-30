/*
resource "awscc_sagemaker_mlflow_tracking_server" "apg_mlflow_sage" {
    artifact_store_uri = "s3://my-unique-bucket-name-test-zzz/art/"
    role_arn = "arn:aws:iam::828238096174:role/apg-role-sage"
    tracking_server_name = "apg-mlflow-sage"
}
*/

/*
resource "awscc_sagemaker_mlflow_tracking_server" "apg_mlflow_sage" {
    artifact_store_uri = "${var.bucket_artifacts}/models/"
    role_arn = aws_iam_role.apg_role_sage.arn
    tracking_server_name = "apg-mlflow-sage"
}
*/