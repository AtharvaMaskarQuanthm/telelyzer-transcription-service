FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .

# Install cuDNN and cublas for CTranslate2
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir nvidia-cudnn-cu12 nvidia-cublas-cu12 && \
    pip install --no-cache-dir -r requirements.txt

# Set library paths for cuDNN
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

# Copy application code
COPY app/ ./app/
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]