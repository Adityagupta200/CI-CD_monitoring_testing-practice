terraform {
    required_providers {
        aws = {
            source = 'hashicorp/aws'
            version= "~> 5.0"
        }
    }
}

provider "aws" {
    region = var.aws_region
}

# Create VPC
resource "aws_vpc" "main" {
    cidr_block = "10.0.0.0/16"
    enable_dns_hostnames = true
    enable_dns_support = true

    tags = {
        Name = "main-vpc"
        Environment = var.Environment
    }
}

# Create Internet Gateway
resource "aws_internet_gateway" "main" {
    vpc_id = aws_vpc.main.id

    tags = {
        Name = "main-igw"
    }
}

# Create public subnet
resource "aws_subnet" "public" {
    vpc_id = aws_vpc.main.id
    cidr_block = "10.0.1.0/24"
    availability_zone = data.aws_availability_zones.available.names[0]
    map_public_ip_on_launch = true

    tags = {
        Name = "public-subnet" 
    }
}

# Create security group
resource "aws_security_group" "app" {
    name_prefix = "app-sg"
    vpc_id = aws_vpc.main.id

    ingress {
        from_port = 80
        to_port = 80
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }

    egress {
        from_port = 0
        to_port = 0
        protocol = "-1"
        cidr_blocks = ["0.0.0.0/0"]
    }

    tags = {
        Name = "app-security-group"
    }
}

# Create EC2 instance for ML model
resource "aws_instance" "ml_server" {
    ami = data.aws_ami.ubuntu.id
    instance_type = "t3.medium"
    subnet_id = aws_subnet.public.id
    vpc_security_group_ids = [aws_security_group.app.id]
    key_name = var.key_pair_name

    user_data = <<-EOF
                   #!/bin/bash
                   apt-get update
                   apt-get install -y docker.io
                   systemctl start docker
                   systemctl enable docker
                   usermode -aG docker ubuntu

                   # Pull and run your ML model container
                   docker pull ghcr.io/your-username/ml-model:latest
                   docker run -d -p 8000:8000 ghcr.io/your-username/ml-model:latest
                   EOF
    tags = {
        Name = "ml-model-server"
        Type = "application"
    }
}

# Data sources
data "aws_availability_zones" "available" {
    state = "available" 
}

data "aws_ami" "ubuntu" {
    most_recent = true
    owners = ["099720109477"]

    filter {
        name = "name"
        values = ["ubuntu/images/hvm-ssd/ubuntu-20.04-amd64-server-*"]
    }
}

