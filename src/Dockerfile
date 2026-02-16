FROM python:3.10-slim

# Set working directory
WORKDIR /app

# --------------------------------------------------
# Install system dependencies
# --------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1\
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Upgrade pip
# --------------------------------------------------
RUN pip install --upgrade pip

# --------------------------------------------------
# Copy requirements first (better caching)
# --------------------------------------------------
COPY requirements.txt .

# --------------------------------------------------
# Install ALL dependencies from requirements.txt
# --------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------
# Copy backend source code
# --------------------------------------------------
COPY . .

# --------------------------------------------------
# Expose API port
# --------------------------------------------------
EXPOSE 8000
ENV PYTHONPATH=/app

# --------------------------------------------------
# Start FastAPI server
# --------------------------------------------------
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]



