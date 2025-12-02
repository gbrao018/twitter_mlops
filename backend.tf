terraform {
  backend "s3" {
    bucket         = "datagrooves-terraform-state-bucket"
    key            = "ebay_mlops/main/terraform.tfstate"
    region         = "us-east-1"
    # This is the key line telling Terraform which profile to use
    profile        = "terraform_admin" 
    encrypt        = true
    dynamodb_table = "terraform-state-locks" 
  }
}
