variable "region" {
  description = "AWS region for all resources."
  default = "us-east-1"
}

variable "git_user" {
  default = "apg-github-actions-user"
}

variable "bucket_data" {
  default = "apg-bucket-data"
}

variable "bucket_logs" {
  default = "apg-bucket-log"
}

variable "bucket_artifacts" {
  default = "apg-bucket-art"
}

variable "role_sage" {
  default = "apg-role-sage"
}

variable "sage_domain" {
  default = "apg-sage-domain"
}

variable "sage_model_grp" {
  default = "apg-model-pkg-heart"
}

/*
variable "tags" {
 type = list(object({
   project = string
   env  = string
 }))
 description = "Project Tags for MLOPS resources"
 default = [{
   project = "APG"
   env  = "NONPROD"
 }]
}
*/

variable "tags" {
  type    = map(string)
  default = {
    project = "APG"
    env     = "NONPRD"
  }

  description = "Extra tags to be included when tagging objects."
}
