resource "aws_sagemaker_domain" "apg_sage_domain" {
	domain_name = var.sage_domain
	auth_mode = "IAM"
	vpc_id = aws_default_vpc.apg_sage_vpc.id
	subnet_ids = [aws_default_subnet.apg_sage_subnet.id]
  #tags = var.tags
	
	default_space_settings {
    		execution_role = aws_iam_role.apg_role_sage.arn
 		}

	default_user_settings {
			execution_role = aws_iam_role.apg_role_sage.arn

            jupyter_server_app_settings {
            lifecycle_config_arns = []

            default_resource_spec {
                    instance_type       = "system"
                    sagemaker_image_arn = "arn:aws:sagemaker:us-east-1:081325390199:image/jupyter-server-3"
                }
            }
		}


}



resource "aws_sagemaker_user_profile" "apg_user_profile_sage" {
	domain_id = aws_sagemaker_domain.apg_sage_domain.id
	user_profile_name = "apg-user-profile-sage"
	user_settings {
		execution_role = aws_iam_role.apg_role_sage.arn
	}
}



/* coments spaces
resource "aws_sagemaker_space" "apg_space_sage" {
  domain_id  = aws_sagemaker_domain.apg_sage_domain.id
  space_name = "apg-space-lab"

  ownership_settings {
    owner_user_profile_name = aws_sagemaker_user_profile.apg_user_profile_sage.user_profile_name
  }

  space_settings {
    app_type = "JupyterLab"

    jupyter_lab_app_settings {
      default_resource_spec {
        instance_type        = "ml.t3.medium"
        #lifecycle_config_arn = aws_sagemaker_studio_lifecycle_config.auto_shutdown.arn
      }
    }
  }

  space_sharing_settings {
    sharing_type = "Shared"
  }
}
*/

resource "aws_sagemaker_model_package_group" "apg_model_pkg_heart" {
  model_package_group_name = var.sage_model_grp
  model_package_group_description = "The model group for Heart Disease"
  tags = var.tags
}