# Fleet AI Oversight Environment
# Meta Hackathon Finals | Team HackWithPals
# OpenEnv Round 2 | Theme 3.1 + Theme 2

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create required directories
RUN mkdir -p plots data checkpoints ui/static

# Run dataset setup
RUN python data/setup_dataset.py

# Generate plots in simulation mode for training results page
RUN python fleet_train.py --simulate --episodes 30 --task-id easy_fleet --mode real || true

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
