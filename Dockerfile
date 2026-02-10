FROM python:3.10-slim

# System deps (needed for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first (important)
RUN pip install --upgrade pip

# 🔥 Install torch CPU explicitly (THIS is the fix)
RUN pip install torch==2.9.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]


