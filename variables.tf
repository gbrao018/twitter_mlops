variable "aws_profile" {
  description = "The AWS profile name configured in your local ~/.aws/credentials file."
  type        = string
  # Sets a default value, which can be overridden by a .tfvars file or command line
  default     = "terraform_admin" 
}

variable "aws_region" {
  description = "The AWS region to deploy to"
  default     = "us-east-1" # Change this to your preferred region
}

variable "app_name" {
  description = "Prefix for all resources"
  default     = "sentiment-analysis"
}

variable "container_image" {
  description = "The Docker image to deploy"
  default     = "ganjidocker/sentiment-analysis-service:latest"
}

variable "container_port" {
  description = "The port the container exposes"
  default     = 8080
}

variable "container_cpu" {
  description = "Fargate task CPU units (e.g., 256, 512, 1024)"
  default     = 1024
}

variable "container_memory" {
  description = "Fargate task memory in MB (e.g., 512, 1024, 2048)"
  default     = 2048
}

variable "local_model_path_in_image" {
  description = "Model is baked into the inference image"
  default = "/app/model"
}