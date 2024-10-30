resource "aws_iam_user_policy_attachment" "apg_git_attach" {
  user       = var.git_user
  policy_arn = aws_iam_policy.apg_super_policy.arn
}
/*
resource "aws_iam_user_policy_attachment" "apg_git_attach" {
  user       = "apg-github-actions-user"
  policy_arn = aws_iam_policy.apg_super_policy.arn
}
*/