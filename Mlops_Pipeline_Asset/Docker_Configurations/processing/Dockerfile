# Dockerfile.processing
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

COPY processing.py .

CMD ["python", "processing.py"]

