#------------------------old 16/6/25 -----------------
#FROM python:3.9-slim

#RUN apt-get update && apt-get install -y \
#    build-essential \
#    curl \
#    && rm -rf /var/lib/apt/lists/*

#WORKDIR /opt/ml/code

#RUN apt-get update && apt-get install -y build-essential

## Install dependencies
#COPY requirements.txt .
#RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

## Set up the SageMaker training environment
##ENV SAGEMAKER_PROGRAM train.py
#COPY train.py /opt/ml/code/train.py


## SageMaker will automatically use the ENTRYPOINT to start training
##ENTRYPOINT ["python", "train.py"]

#ENV SAGEMAKER_PROGRAM train.py

#------------------------------------new---------------


FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# Ensure pip is up-to-date
RUN pip install --upgrade pip

# Uninstall any old xgboost if present
RUN pip uninstall -y xgboost || true

# Install xgboost first to avoid version conflicts
RUN pip install --no-cache-dir "xgboost>=1.3.0"

# Now install the rest of the requirements
COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Copy training script
COPY train.py /opt/ml/code/train.py

# Set the script to run by SageMaker
ENV SAGEMAKER_PROGRAM train.py



