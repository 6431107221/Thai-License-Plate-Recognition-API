FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY src/ ./src/
COPY ocr_minimal/ ./ocr_minimal/


EXPOSE 8080

CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8080"]