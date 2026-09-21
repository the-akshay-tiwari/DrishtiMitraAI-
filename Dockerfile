# DrishtiMitra Production Multi-Stage Deployment Dockerfile

# Stage 1: Build Frontend static bundle
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python FastAPI inference server (CPU Optimized)
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install lightweight CPU-only PyTorch wheel (~180MB instead of 3GB CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python backend dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend code, compiled frontend bundle, and model artifacts
COPY backend/ /app/backend/
COPY --from=frontend-builder /app/dist /app/dist
COPY artifacts/ /app/artifacts/
COPY artifacts_output/ /app/artifacts_output/
RUN ls -lh /app/artifacts_output/v3_4/models/matlab_v3_4_full_weights.mat

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
