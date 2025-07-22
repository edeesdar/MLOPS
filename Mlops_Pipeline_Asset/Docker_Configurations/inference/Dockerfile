#FROM python:3.9-slim

# Install OS dependencies
#RUN apt-get update && apt-get install -y \
 #   build-essential \
  #  curl \
   # && rm -rf /var/lib/apt/lists/*

# Set working directory
#WORKDIR /opt/ml/code

# Install Python dependencies
#COPY requirements.txt .
#RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt




# Copy your inference code
#COPY inference.py /opt/ml/code/inference.py

# Set environment variables for SageMaker Batch
#ENV SAGEMAKER_PROGRAM=inference.py
#ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
#ENV PYTHONUNBUFFERED=TRUE


# This gets ignored in Batch Transform (it's executed via SageMaker)
#CMD ["python", "inference.py"]
#ENV SAGEMAKER_PROGRAM inference.py

#---------------------------------------new------------------------------------------

# Start from a base Python image
FROM python:3.9-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV AWS_DEFAULT_REGION=us-east-1

# Create a working directory
WORKDIR /opt/program

# Install required Python packages
COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Copy your inference script and any other code
COPY inference.py .

# Define the ENTRYPOINT as required by SageMaker Batch Transform
ENTRYPOINT ["python", "inference.py"]

#----------------------------------------------------------------------------------------------


#FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1-cpu-py3
#FROM amazonlinux:2

#RUN apt-get update && apt-get install -y \
 #   build-essential \
  #  curl \
   # && rm -rf /var/lib/apt/lists/*
#RUN yum update -y && \
#    yum install -y \
#    gcc \
#    gcc-c++ \
#    make \
#    curl \
#    python3 \
#    python3-pip && \
#    yum clean all

# Install additional libraries you need (e.g., pandas, sklearn)
#COPY requirements.txt .
#RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Set the working directory
#WORKDIR /opt/ml/code

# Copy the inference.py script to the container
#COPY inference.py /opt/ml/code/inference.py

# Set environment variables (if any)
#ENV PYTHONUNBUFFERED=TRUE

# SageMaker provides default directories for model artifacts and temporary files
#ENV SAGEMAKER_PROGRAM=inference.py
#ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

