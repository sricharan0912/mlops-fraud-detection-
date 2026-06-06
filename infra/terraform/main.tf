terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # Remote state — configure your bucket/key before applying
  # backend "s3" {
  #   bucket = "your-tfstate-bucket"
  #   key    = "fraud-platform/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
}

# ── VPC ──────────────────────────────────────────────────────────────────────
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = { Name = "${var.project}-vpc" }
}

resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags = { Name = "${var.project}-public-${count.index}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags = { Name = "${var.project}-private-${count.index}" }
}

data "aws_availability_zones" "available" { state = "available" }

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
}

resource "aws_eip" "nat" { domain = "vpc" }

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  depends_on    = [aws_internet_gateway.igw]
}

# ── ECS Fargate ───────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "main" {
  name = "${var.project}-cluster"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecr_repository" "api" {
  name                 = var.ecr_repository_name
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${var.project}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "fraud-api"
      image = "${aws_ecr_repository.api.repository_url}:latest"
      portMappings = [{ containerPort = 8000, protocol = "tcp" }]
      environment = [
        { name = "USE_MLFLOW_REGISTRY", value = "true" },
      ]
      secrets = [
        { name = "DATABASE_URL",         valueFrom = aws_secretsmanager_secret.db_url.arn },
        { name = "MLFLOW_TRACKING_URI",  valueFrom = aws_secretsmanager_secret.mlflow_uri.arn },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/${var.project}-api"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }
    }
  ])
}

resource "aws_ecs_service" "api" {
  name            = "${var.project}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "fraud-api"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.http]
}

# ── RDS Postgres ──────────────────────────────────────────────────────────────
resource "aws_db_instance" "postgres" {
  identifier             = "${var.project}-db"
  engine                 = "postgres"
  engine_version         = "16"
  instance_class         = var.db_instance_class
  allocated_storage      = 20
  db_name                = "fraud"
  username               = "fraud_user"
  password               = var.db_password
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  skip_final_snapshot    = true
  deletion_protection    = false
  storage_encrypted      = true
  tags = { Name = "${var.project}-postgres" }
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.project}-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id
}

# ── S3 (MLflow artifacts) ──────────────────────────────────────────────────────
resource "aws_s3_bucket" "artifacts" {
  bucket        = var.s3_artifact_bucket
  force_destroy = false
  tags          = { Name = "${var.project}-artifacts" }
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    id     = "glacier-after-90-days"
    status = "Enabled"
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    filter { prefix = "mlflow-artifacts/" }
  }
}

# ── Secrets Manager ────────────────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "db_url" {
  name = "${var.project}/database-url"
}

resource "aws_secretsmanager_secret" "mlflow_uri" {
  name = "${var.project}/mlflow-tracking-uri"
}

# ── ALB ───────────────────────────────────────────────────────────────────────
resource "aws_lb" "main" {
  name               = "${var.project}-alb"
  internal           = false
  load_balancer_type = "application"
  subnets            = aws_subnet.public[*].id
  security_groups    = [aws_security_group.alb.id]
}

resource "aws_lb_target_group" "api" {
  name        = "${var.project}-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"
  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 30
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# ── IAM ───────────────────────────────────────────────────────────────────────
resource "aws_iam_role" "ecs_task_execution" {
  name = "${var.project}-ecs-execution-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  name = "${var.project}-ecs-task-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "s3-mlflow-artifacts"
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:GetObject", "s3:ListBucket"], Resource = [
        aws_s3_bucket.artifacts.arn,
        "${aws_s3_bucket.artifacts.arn}/*",
      ]}
    ]
  })
}

# ── Security Groups ────────────────────────────────────────────────────────────
resource "aws_security_group" "alb" {
  name   = "${var.project}-alb-sg"
  vpc_id = aws_vpc.main.id
  ingress { from_port = 80, to_port = 80, protocol = "tcp", cidr_blocks = ["0.0.0.0/0"] }
  egress  { from_port = 0,  to_port = 0,  protocol = "-1", cidr_blocks = ["0.0.0.0/0"] }
}

resource "aws_security_group" "api" {
  name   = "${var.project}-api-sg"
  vpc_id = aws_vpc.main.id
  ingress { from_port = 8000, to_port = 8000, protocol = "tcp", security_groups = [aws_security_group.alb.id] }
  egress  { from_port = 0,    to_port = 0,    protocol = "-1", cidr_blocks = ["0.0.0.0/0"] }
}

resource "aws_security_group" "rds" {
  name   = "${var.project}-rds-sg"
  vpc_id = aws_vpc.main.id
  ingress { from_port = 5432, to_port = 5432, protocol = "tcp", security_groups = [aws_security_group.api.id] }
}
