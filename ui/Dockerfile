FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip FIRST
RUN pip install --upgrade pip

# Copy project
COPY . .

# Install python dependencies
RUN pip install --no-cache-dir --upgrade \
    streamlit \
    requests \
    pillow \
    numpy

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
