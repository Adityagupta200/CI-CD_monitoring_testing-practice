# # Use an official Python runtime as the base image
# FROM python:3.9-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy requirements.txt first for better caching
# COPY requirements.txt .

# # Install any needed dependencies specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application code
# COPY . .

# # Make port 8080 available to the world outside this container
# EXPOSE 8080

# # Define environment variable
# ENV FLASK_APP=main.py

# # Run main.py when the container launches
# CMD ["python", "main.py"]

"""Multi-stage Dockerfile for optimized container"""
# FROM python:3.9-slim as builder

# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# FROM python:3.9-slim as runtime

# WORKDIR /app
# COPY --from=builder /usr/local/lib/python3.9/site-packages
# COPY . .

# EXPOSE 8080
# CMD["python", "app.py"]

"""Dockerfile for ML model serving"""
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY app.py .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app
USER app

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1


