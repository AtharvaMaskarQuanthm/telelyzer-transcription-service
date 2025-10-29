FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies including specific cuDNN version
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    ca-certificates \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12=9.5.1.17-1 \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

# Find and set cuDNN library path
RUN CUDNN_PATH=$(find /usr -name "libcudnn_ops.so.9" -exec dirname {} \; | head -n 1) && \
    echo "Found cuDNN at: $CUDNN_PATH" && \
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$CUDNN_PATH" >> /etc/profile.d/cudnn.sh

# Set the library path for this session
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]