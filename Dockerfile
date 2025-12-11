# ถ้าจะใช้ GPU บน GCE ต้องเปลี่ยน Base Image เป็น nvidia/cuda
FROM python:3.11-slim

WORKDIR /app

# Install System Dependencies
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libgl1 && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --extra-index-url https://pypi.org/simple


COPY src/ ./src/
COPY ocr_minimal/ ./ocr_minimal/


EXPOSE 8080


CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8080"]