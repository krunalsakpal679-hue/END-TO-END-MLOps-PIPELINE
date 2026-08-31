# Use an official Python runtime as a parent image
FROM python:3.11-slim@sha256:1042b61448fef4ba92d16a8c7eb4996d027568ce64792a7877fd88511e0af7c6

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# build-essential, gcc, g++: needed for compiling native extensions
# curl: required for the HEALTHCHECK
# libgomp1: required for scikit-learn/numpy runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies directly from requirements.txt
# Using --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Security: Create and configure non-root user
RUN groupadd -g 1001 mlserve && \
    useradd -u 1001 -g mlserve -s /bin/bash -m mlserve && \
    chown -R mlserve:mlserve /app

# Switch to the non-root user
USER mlserve

# Set environment variables
ENV MODEL_PATH="models/champion_model.pkl" \
    MLFLOW_TRACKING_URI="http://localhost:5000" \
    PORT=8000 \
    LOG_LEVEL="INFO" \
    PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 8000

# Configure healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
