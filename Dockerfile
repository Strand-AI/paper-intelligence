# Paper Intelligence MCP Server
# Multi-stage build for smaller image

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY paper_intelligence/ paper_intelligence/

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install .

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies for marker (OCR, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY paper_intelligence/ paper_intelligence/

# Create directories for data
RUN mkdir -p /data/input /data/output

# Set environment variables
ENV PAPER_INTELLIGENCE_OUTPUT=/data/output
ENV PYTHONUNBUFFERED=1
# Use CPU in Docker (MPS not available in containers)
ENV TORCH_DEVICE=cpu

# Expose port for HTTP transport (optional)
EXPOSE 8080

# Default to stdio transport for MCP
ENTRYPOINT ["python", "-m", "paper_intelligence.server"]
