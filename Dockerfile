# Use lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Install system dependencies for OpenCV and builds
# Clean up apt list to save space
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and weights
COPY src/ ./src/
COPY weights/ ./weights/

# Expose port
EXPOSE 8080

# Run the application with uvicorn
CMD ["sh", "-c", "uvicorn src.api_server:app --host 0.0.0.0 --port ${PORT}"]