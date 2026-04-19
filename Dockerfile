# AllSkyAnalyzer – Docker image
#
# Base image: NVIDIA CUDA 12.4 + cuDNN on Ubuntu 22.04.
# The image bundles Python 3.11, OpenCV (with CUDA), PyTorch, Astropy,
# and an optional astrometry.net installation.
#
# Build:
#   docker build -t allskyanalyzer:latest .
#
# Run (single image):
#   docker run --gpus all \
#       -v /var/lib/indi-allsky:/var/lib/indi-allsky:ro \
#       -v /data/output:/data/output \
#       allskyanalyzer:latest analyze /var/lib/indi-allsky/images/latest.jpg
#
# Run (watcher daemon):
#   docker run --gpus all \
#       -v /var/lib/indi-allsky:/var/lib/indi-allsky:ro \
#       -v /data/output:/data/output \
#       allskyanalyzer:latest watch
#
# Run (REST API):
#   docker run --gpus all -p 8000:8000 \
#       -v /var/lib/indi-allsky:/var/lib/indi-allsky:ro \
#       -v /data/output:/data/output \
#       allskyanalyzer:latest serve

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# --------------------------------------------------------------------------
# System packages
# --------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-venv \
        # OpenCV runtime dependencies
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgomp1 \
        # SSH client (paramiko uses system libssl)
        libssl3 \
        # astrometry.net CLI tool + default index files (optional but recommended)
        astrometry.net \
        astrometry-data-tycho2 \
        # Other utilities
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# --------------------------------------------------------------------------
# Python dependencies
# --------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 wheels, then the rest of the requirements.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------------
# Application code
# --------------------------------------------------------------------------
COPY . .

# --------------------------------------------------------------------------
# Runtime configuration
# --------------------------------------------------------------------------
# Allow the config path to be overridden at runtime.
ENV ALLSKY_CONFIG=/app/config/config.yaml

# Output + settings directories (can be volume-mounted).
RUN mkdir -p /data/output /data/settings
VOLUME ["/var/lib/indi-allsky", "/data/output", "/data/settings"]

EXPOSE 8000

ENTRYPOINT ["python", "main.py", "--config", "/app/config/config.yaml"]
CMD ["watch"]
