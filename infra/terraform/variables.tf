variable "aws_region" {
  default = "us-east-1"
}

variable "project" {
  default = "fraud-platform"
}

variable "environment" {
  default = "prod"
}

variable "db_instance_class" {
  default = "db.t4g.micro"
}

variable "db_password" {
  description = "RDS Postgres password (stored in Secrets Manager)"
  sensitive   = true
}

variable "ecr_repository_name" {
  default = "fraud-api"
}

variable "ecs_task_cpu" {
  default = "512"
}

variable "ecs_task_memory" {
  default = "1024"
}

variable "s3_artifact_bucket" {
  description = "S3 bucket for MLflow artifacts"
}
