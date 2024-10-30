/*
resource "aws_s3_bucket" "data_bucket" {
  bucket = "apg-data-bucket"
  tags = {
    Name        = "MyBucket"
    Environment = "Dev"
  }
} 
*/

resource "aws_s3_bucket" "data_bucket" {
  bucket = var.bucket_data
  tags = merge(var.tags , {
    ResourceName = "APG-Data-Bucket"
  })
  #tags = concat(var.tags , {
  #  ResourceName = "APG-Data-Bucket"
  #})
}

resource "aws_s3_bucket" "logs_bucket" {
  bucket = var.bucket_logs
  tags = var.tags
}

resource "aws_s3_bucket" "artifacts_bucket" {
  bucket = var.bucket_artifacts
  tags = var.tags
}

