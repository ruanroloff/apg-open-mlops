/*
resource "aws_iam_policy" "apg_super_policy" {
  name        = "APG-Super-Policy-Sage"
  path        = "/"
  description = "Sagemaker super policy"
  tags = var.tags

  # Terraform's "jsonencode" function converts a
  # Terraform expression result to valid JSON syntax.
  policy = jsonencode({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "VisualEditor0",
                "Effect": "Allow",
                "Action": [
                    "sagemaker:CreateUserProfile",
                    "sagemaker:DescribeModelPackage",
                    "sagemaker:ListSharedModelVersions",
                    "sagemaker:DescribeModelPackageGroup",
                    "sagemaker:GetRecord",
                    "sagemaker:DescribeFlowDefinition",
                    "sagemaker:DescribeAlgorithm",
                    "sagemaker:DescribeAutoMLJobV2",
                    "sagemaker:GetScalingConfigurationRecommendation",
                    "sagemaker:DescribeTransformJob",
                    "sagemaker:DescribeInferenceRecommendationsJob",
                    "sagemaker:DescribeHumanLoop",
                    "sagemaker:ListClusterNodes",
                    "sagemaker:BatchDescribeModelPackage",
                    "sagemaker:DescribeSharedModel",
                    "sagemaker:DescribeDeviceFleet",
                    "sagemaker:DescribeOptimizationJob",
                    "sagemaker:DescribeHyperParameterTuningJob",
                    "sagemaker:DescribeWorkforce",
                    "sagemaker:DescribeSpace",
                    "sagemaker:DescribeProcessingJob",
                    "sagemaker:GetDeviceFleetReport",
                    "sagemaker:DescribeStudioLifecycleConfig",
                    "sagemaker:DescribeImageVersion",
                    "sagemaker:ListPipelineParametersForExecution",
                    "sagemaker:DescribeInferenceComponent",
                    "sagemaker:DescribeHumanTaskUi",
                    "sagemaker:GetDeployments",
                    "sagemaker:DescribeProject",
                    "sagemaker:ListImageVersions",
                    "sagemaker:ListModelCardExportJobs",
                    "sagemaker:ListHubContents",
                    "sagemaker:DescribeModelExplainabilityJobDefinition",
                    "sagemaker:DescribeCluster",
                    "sagemaker:DescribeEndpoint",
                    "sagemaker:DescribeUserProfile",
                    "sagemaker:InvokeEndpoint",
                    "sagemaker:DescribeFeatureMetadata",
                    "sagemaker:DescribeEdgePackagingJob",
                    "sagemaker:DescribeFeatureGroup",
                    "sagemaker:DescribeModelQualityJobDefinition",
                    "sagemaker:DescribeMlflowTrackingServer",
                    "sagemaker:DescribeModel",
                    "sagemaker:CreateDomain",
                    "sagemaker:DescribePipeline",
                    "sagemaker:DescribeArtifact",
                    "sagemaker:CreateCluster",
                    "sagemaker:DescribeImage",
                    "sagemaker:InvokeEndpointAsync",
                    "sagemaker:CreateArtifact",
                    "sagemaker:DescribePipelineDefinitionForExecution",
                    "sagemaker:DescribeTrialComponent",
                    "sagemaker:DescribeClusterNode",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:DescribeLabelingJob",
                    "sagemaker:DescribeDataQualityJobDefinition",
                    "sagemaker:DescribeInferenceExperiment",
                    "sagemaker:ListMlflowTrackingServers",
                    "sagemaker:ListHubContentVersions",
                    "sagemaker:DescribeApp",
                    "sagemaker:ListLabelingJobsForWorkteam",
                    "sagemaker:InvokeEndpointWithResponseStream",
                    "sagemaker:ListModelCardVersions",
                    "sagemaker:ListPipelineExecutions",
                    "sagemaker:DescribeAction",
                    "sagemaker:DescribeModelCardExportJob",
                    "sagemaker:DescribeSubscribedWorkteam",
                    "sagemaker:DescribeAutoMLJob",
                    "sagemaker:ListTrainingJobsForHyperParameterTuningJob",
                    "sagemaker:DescribeEndpointConfig",
                    "sagemaker:BatchGetRecord",
                    "sagemaker:GetDeviceRegistration",
                    "sagemaker:DescribeNotebookInstance",
                    "sagemaker:DescribeAppImageConfig",
                    "sagemaker:DescribeNotebookInstanceLifecycleConfig",
                    "sagemaker:DescribeModelCard",
                    "sagemaker:DescribeTrial",
                    "sagemaker:DescribeContext",
                    "sagemaker:DescribeEdgeDeploymentPlan",
                    "sagemaker:DescribeHubContent",
                    "sagemaker:DescribeMonitoringSchedule",
                    "sagemaker:DeleteDomain",
                    "sagemaker:ListTags",
                    "sagemaker:GetModelPackageGroupPolicy",
                    "sagemaker:DescribePipelineExecution",
                    "sagemaker:DescribeWorkteam",
                    "sagemaker:ListModelPackages",
                    "sagemaker:DescribeModelBiasJobDefinition",
                    "sagemaker:BatchGetMetrics",
                    "sagemaker:DescribeCompilationJob",
                    "sagemaker:DescribeExperiment",
                    "sagemaker:DescribeHub",
                    "sagemaker:ListAliases",
                    "sagemaker:DescribeDomain",
                    "sagemaker:DescribeCodeRepository",
                    "sagemaker:ListPipelineExecutionSteps",
                    "sagemaker:DescribeDevice",
                    "sagemaker:DeleteUserProfile",
                    "sagemaker:UpdateUserProfile",
                    "sagemaker:ListUserProfiles",
                    "iam:CreateServiceLinkedRole",
                    "iam:PassRole",
                    "sagemaker-mlflow:*"
                ],
                "Resource": [
                    "arn:aws:sagemaker:*:*:domain/*",
                    "arn:aws:sagemaker:*:*:user-profile/*"
                ]
            },
            {
                "Sid": "VisualEditor1",
                "Effect": "Allow",
                "Action": [
                    "sagemaker:ListApps",
                    "sagemaker:ListArtifacts",
                    "sagemaker:ListCandidatesForAutoMLJob",
                    "sagemaker:ListModelBiasJobDefinitions",
                    "sagemaker:ListTransformJobs",
                    "sagemaker:ListHumanTaskUis",
                    "sagemaker:GetResourcePolicy",
                    "sagemaker:ListMonitoringExecutions",
                    "sagemaker:Search",
                    "sagemaker:ListDataQualityJobDefinitions",
                    "sagemaker:ListMonitoringAlertHistory",
                    "sagemaker:ListTrainingJobs",
                    "sagemaker:GetLineageGroupPolicy",
                    "sagemaker:ListExperiments",
                    "sagemaker:ListSubscribedWorkteams",
                    "sagemaker:ListFeatureGroups",
                    "sagemaker:ListClusters",
                    "sagemaker:ListInferenceExperiments",
                    "sagemaker:ListLineageGroups",
                    "sagemaker:ListAlgorithms",
                    "sagemaker:ListNotebookInstanceLifecycleConfigs",
                    "sagemaker:ListDeviceFleets",
                    "sagemaker:ListCompilationJobs",
                    "sagemaker:ListTrials",
                    "sagemaker:ListOptimizationJobs",
                    "sagemaker:ListEndpointConfigs",
                    "sagemaker:ListActions",
                    "sagemaker:ListStudioLifecycleConfigs",
                    "sagemaker:RenderUiTemplate",
                    "sagemaker:ListModelExplainabilityJobDefinitions",
                    "sagemaker:ListModelCards",
                    "sagemaker:ListDomains",
                    "sagemaker:ListEdgePackagingJobs",
                    "sagemaker:ListModelMetadata",
                    "sagemaker:ListUserProfiles",
                    "sagemaker:ListAppImageConfigs",
                    "sagemaker:ListStageDevices",
                    "sagemaker:ListWorkteams",
                    "sagemaker:ListResourceCatalogs",
                    "sagemaker:GetSagemakerServicecatalogPortfolioStatus",
                    "sagemaker:ListSharedModels",
                    "sagemaker:ListProjects",
                    "sagemaker:ListContexts",
                    "sagemaker:DescribeLineageGroup",
                    "sagemaker:ListAutoMLJobs",
                    "sagemaker:ListHumanLoops",
                    "sagemaker:ListMonitoringSchedules",
                    "sagemaker:ListInferenceRecommendationsJobSteps",
                    "sagemaker:ListProcessingJobs",
                    "sagemaker:QueryLineage",
                    "sagemaker:ListAssociations",
                    "sagemaker:ListEdgeDeploymentPlans",
                    "sagemaker:ListSharedModelEvents",
                    "sagemaker:ListModelPackageGroups",
                    "sagemaker:ListImages",
                    "sagemaker:ListDevices",
                    "sagemaker:ListInferenceRecommendationsJobs",
                    "sagemaker:ListModelQualityJobDefinitions",
                    "sagemaker:ListNotebookInstances",
                    "sagemaker:ListFlowDefinitions",
                    "sagemaker:ListTrialComponents",
                    "sagemaker:ListHubs",
                    "sagemaker:ListInferenceComponents",
                    "sagemaker:ListHyperParameterTuningJobs",
                    "sagemaker:ListLabelingJobs",
                    "sagemaker:ListWorkforces",
                    "sagemaker:GetSearchSuggestions",
                    "sagemaker:ListSpaces",
                    "sagemaker:ListMonitoringAlerts",
                    "sagemaker:ListPipelines",
                    "sagemaker:ListModels",
                    "sagemaker:ListEndpoints",
                    "sagemaker:ListCodeRepositories"
                ],
                "Resource": "*"
            }
        ]
    }
  )
}
*/
resource "aws_iam_policy" "apg_mlflow_policy" {
  name        = "APG-Policy-2-Mlflow"
  path        = "/"
  description = "Sagemaker policy for MLflow"
  tags = var.tags

  # Terraform's "jsonencode" function converts a
  # Terraform expression result to valid JSON syntax.
  policy = jsonencode({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "VisualEditor0",
                "Effect": "Allow",
                "Action": [
                    "sagemaker-mlflow:AccessUI",
                    "sagemaker-mlflow:*"
                ],
                "Resource": "*"
            },
            {
                "Sid": "VisualEditor1",
                "Effect": "Allow",
                "Action": "sagemaker-mlflow:*",
                "Resource": "arn:aws:sagemaker:*:*:mlflow-tracking-server/*"
            }
        ]
    }
  )
}
