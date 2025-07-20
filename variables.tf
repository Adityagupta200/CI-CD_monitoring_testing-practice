variable "aws_region" {
    description = "AWS region"
    type = string
    default = "us-west-2"
}

variable "environment" {
    description = "Environment name"
    type = string
    default = "development"
}

variable "key_pair_name" {
    description = "Name of the AWS key pair"
    type = string
}