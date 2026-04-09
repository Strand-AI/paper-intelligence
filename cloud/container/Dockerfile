# Marker PDF converter for Cloudflare Containers
# Target: linux/amd64 (CF Containers requirement)

FROM python:3.11-slim

WORKDIR /app

# System deps for Marker (OCR, image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (much smaller than full torch)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install Marker + server deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Marker models into the image (avoids cold-start download)
RUN python -c "from marker.models import create_model_dict; create_model_dict()" \
    || echo "Model pre-download failed — will download on first request"

COPY server.py .

ENV PYTHONUNBUFFERED=1
ENV TORCH_DEVICE=cpu

EXPOSE 8080

# Use gunicorn with long timeout for large PDFs
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "--workers", "1", "server:app"]
