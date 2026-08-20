# Multi-Stage Build for Latent Video Prediction Studio

# Stage 1: Build React Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# Stage 2: Production FastAPI + PyTorch Backend
FROM python:3.10-slim AS backend
WORKDIR /app

# Install minimal OS dependencies for video rendering & healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend codebase & pretrained model checkpoints
COPY config.py .
COPY model/ ./model/
COPY data/ ./data/
COPY training/ ./training/
COPY api/ ./api/
COPY checkpoints/ ./checkpoints/
COPY runs/ ./runs/

# Copy compiled frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose HTTP port
EXPOSE 8000

# Healthcheck monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start production server
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
