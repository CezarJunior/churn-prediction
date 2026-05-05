# ============================================================
#  Churn Prediction — Dockerfile
#  Target: Hugging Face Spaces (free tier) / any Linux host
#
#  Build : docker build -t churn-prediction .
#  Run   : docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... churn-prediction
# ============================================================

FROM python:3.10-slim

# HF Spaces expects the app to listen on port 7860
EXPOSE 7860

# Avoid .pyc files and buffer issues in Docker logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Create a non-root user (good practice; HF Spaces requires it)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system dependencies required by faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY --chown=appuser:appuser . .

# Ensure critical directories exist
RUN mkdir -p data/raw models mlruns

USER appuser

# Pre-download the embedding model so the first request isn't slow.
# This bakes BAAI/bge-small-en-v1.5 (~130 MB) into the image.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5')" || true

# Train the model at build time if no pre-trained artefact is present.
# On HF Spaces the model is trained once during the build and reused.
RUN if [ ! -f models/best_model.joblib ]; then \
        echo 'No pre-trained model found — running training (20 trials) …'; \
        python -m src.pipeline.train --trials 20; \
    else \
        echo 'Pre-trained model found — skipping training.'; \
    fi

CMD ["python", "app.py"]
