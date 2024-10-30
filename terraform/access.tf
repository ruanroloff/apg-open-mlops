resource "aws_iam_role" "apg_role_sage" {
	name = var.role_sage
	path = "/"
	assume_role_policy = data.aws_iam_policy_document.apg_role_sage.json
	managed_policy_arns = ["arn:aws:iam::aws:policy/AmazonSageMakerFullAccess", "arn:aws:iam::aws:policy/AmazonS3FullAccess"]
	tags = var.tags
}

resource "aws_iam_role_policy_attachment" "apg_role_sage" {
  role       = aws_iam_role.apg_role_sage.name
  policy_arn = aws_iam_policy.apg_mlflow_policy.arn
}

data "aws_iam_policy_document" "apg_role_sage" {
	statement {
		actions = ["sts:AssumeRole"]
		principals {
			type = "Service"
			identifiers = ["sagemaker.amazonaws.com"]
		}
	}
}

/*
resource "aws_iam_user_policy_attachment" "apg_git_attach" {
  user       = var.git_user
  policy_arn = aws_iam_policy.apg_super_policy.arn
}
*/

/*
AmazonEC2FullAccess
AmazonS3FullAccess
AmazonSageMakerFullAccess
AWSCloudFormationFullAccess
AWSNetworkManagerFullAccess
IAMFullAccess
ResourceGroupsandTagEditorFullAccess
*/