# =====================================================================
# Stage 1: Builder
# =====================================================================
FROM python:3.11-slim AS builder

# Install build dependencies for compiling Python packages (e.g. numpy, scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements and pre-compile all dependencies into wheels
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt


# =====================================================================
# Stage 2: Runtime
# =====================================================================
FROM python:3.11-slim

# Install runtime dependencies like curl for the HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Security: Create and configure non-root user
RUN groupadd -g 1001 mlserve && \
    useradd -u 1001 -g mlserve -s /bin/bash -m mlserve

WORKDIR /app

# Copy the pre-built wheels from Stage 1
COPY --from=builder /wheels /wheels

# Install the wheels without caching to minimize layer size
RUN pip install --no-cache /wheels/*

# Copy the application payload
COPY app/ /app/app/

# Set sensible environment variable defaults
ENV MODEL_URI="models:/anti_gravity_v1/Production" \
    MLFLOW_TRACKING_URI="http://localhost:5000" \
    LOG_LEVEL="INFO" \
    WORKERS="4" \
    PORT="8080" \
    PYTHONUNBUFFERED="1"

# Expose standard port
EXPOSE 8080

# Configure Liveness Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/v1/ready || exit 1

# Change user ownership securely to the non-root execution profile
RUN chown -R mlserve:mlserve /app
USER mlserve

# Start the uvicorn API securely binding to the port mapped via env vars
ENTRYPOINT ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers $WORKERS"]
