output "alb_dns_name" {
  description = "Public DNS of the Application Load Balancer — your API URL"
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecr_repository_uri" {
  description = "ECR repository URI"
  value       = "154932391114.dkr.ecr.us-east-1.amazonaws.com/fraud-api"
}

output "rds_endpoint" {
  description = "RDS Postgres endpoint"
  value       = aws_db_instance.postgres.endpoint
  sensitive   = true
}

output "ecs_cluster_name" {
  description = "ECS cluster name (use in GitHub Actions deploy secret)"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name (use in GitHub Actions deploy secret)"
  value       = aws_ecs_service.api.name
}

output "s3_artifact_bucket" {
  description = "S3 bucket for MLflow artifacts"
  value       = var.s3_artifact_bucket
}
