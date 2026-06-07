variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "account_id" {
  type    = string
  default = "154932391114"
}

variable "project_name" {
  type    = string
  default = "fraud-platform"
}

variable "environment" {
  type    = string
  default = "prod"
}

variable "ecr_image_uri" {
  type    = string
  default = "154932391114.dkr.ecr.us-east-1.amazonaws.com/fraud-api:latest"
}

variable "s3_artifact_bucket" {
  type    = string
  default = "fraud-platform-artifacts-154932391114"
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "db_password" {
  description = "RDS master password — set via TF_VAR_db_password env var"
  type        = string
  sensitive   = true
}

variable "api_task_cpu" {
  type    = number
  default = 512
}

variable "api_task_memory" {
  type    = number
  default = 1024
}

variable "api_desired_count" {
  type    = number
  default = 1
}
