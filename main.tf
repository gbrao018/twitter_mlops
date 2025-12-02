# ----------------------------------------------------
# 1. VPC and Networking Setup (Minimal for Fargate)
# ----------------------------------------------------

# Create a new VPC
resource "aws_vpc" "app_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = {
    Name = "${var.app_name}-vpc"
  }
}

# Create two public subnets across two different availability zones
resource "aws_subnet" "public_subnet" {
  count                   = 2
  vpc_id                  = aws_vpc.app_vpc.id
  cidr_block              = cidrsubnet(aws_vpc.app_vpc.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.app_name}-public-subnet-${count.index}"
  }
}

# Data source for available AZs
data "aws_availability_zones" "available" {}

# Create an Internet Gateway for public access
resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.app_vpc.id

  tags = {
    Name = "${var.app_name}-igw"
  }
}

# Create a route table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.app_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }

  tags = {
    Name = "${var.app_name}-public-rt"
  }
}

# Associate route table with subnets
resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public_subnet[count.index].id
  route_table_id = aws_route_table.public.id
}

# Security group to allow traffic on the container port (HTTP)
resource "aws_security_group" "app_sg" {
  vpc_id      = aws_vpc.app_vpc.id
  description = "Allow inbound traffic for the sentiment analysis service"

  # Ingress rule: Allow inbound traffic on the exposed container port
  ingress {
    description = "Allow HTTP access"
    from_port   = var.container_port
    to_port     = var.container_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Egress rule: Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.app_name}-sg"
  }
}

# ----------------------------------------------------
# 2. IAM Roles for ECS Fargate
# ----------------------------------------------------

# ECS Task Execution Role (Required by Fargate to pull image, log to CloudWatch)
resource "aws_iam_role" "ecs_execution_role" {
  name = "${var.app_name}-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ----------------------------------------------------
# 3. ECS Cluster and Log Group
# ----------------------------------------------------

# ECS Cluster
resource "aws_ecs_cluster" "app_cluster" {
  name = "${var.app_name}-cluster"
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "app_log_group" {
  name              = "/ecs/${var.app_name}-service"
  retention_in_days = 7
}

# ----------------------------------------------------
# 4. ECS Task Definition (The blueprint)
# ----------------------------------------------------

resource "aws_ecs_task_definition" "app_task" {
  family                   = "${var.app_name}-task"
  cpu                      = var.container_cpu
  memory                   = var.container_memory
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  
  container_definitions = jsonencode([
    {
      name      = var.app_name
      image     = var.container_image
      cpu       = var.container_cpu
      memory    = var.container_memory
      essential = true
      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]
      # Environment variable MODEL_PATH is no longer needed as we use the registry URI
      environment = [] 
      
      # --- CRITICAL FIX: Use MLflow Model Registry URI for MLOps best practice ---
      # This connects the deployment directly to the governance layer (Model Registry)
      
      # --- FIX: Execute mlflow as a Python module to ensure the command is found ---
      #command = [
      #  "python3",         # Use the python interpreter
      #  "-m",             # Run the following as a module
      #  "mlflow",         # The module name
      #  "models",
      #  "serve",
      #  "--model-uri",
      #  "file://${var.local_model_path_in_image}",
      #  "--host",
      #  "0.0.0.0",
      #  "--port",
      #  tostring(var.container_port) 
      #]
      # -----------------------------------------------------------------
      # -----------------------------------------------------------------
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app_log_group.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])

  tags = {
    Name = "${var.app_name}-task"
  }
}

# ----------------------------------------------------
# 5. ECS Service (The orchestrator)
# ----------------------------------------------------

resource "aws_ecs_service" "app_service" {
  name            = "${var.app_name}-service"
  cluster         = aws_ecs_cluster.app_cluster.id
  task_definition = aws_ecs_task_definition.app_task.arn
  launch_type     = "FARGATE"
  desired_count   = 1 # Start with 1 instance
  
  # Networking configuration for Fargate
  network_configuration {
    subnets          = aws_subnet.public_subnet[*].id
    security_groups  = [aws_security_group.app_sg.id]
    assign_public_ip = true
  }

  force_new_deployment = true

  tags = {
    Name = "${var.app_name}-service"
  }
}

# ----------------------------------------------------
# 6. Output the Public URL/IP
# ----------------------------------------------------

# This output will show the public IP of the first running task.
output "service_ip" {
  value = aws_ecs_service.app_service.network_configuration[0].assign_public_ip ? "The service IP address will be dynamically assigned upon deployment." : "N/A"
  description = "The service is deployed in Fargate with a public IP."
}