output "alb_dns_name" {
  description = "Public DNS of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "ecr_repository_url" {
  description = "ECR repository URL for the API image"
  value       = aws_ecr_repository.api.repository_url
}

output "rds_endpoint" {
  description = "RDS Postgres endpoint"
  value       = aws_db_instance.postgres.endpoint
  sensitive   = true
}

output "s3_artifact_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}
