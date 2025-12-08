# 1. Base Image: ใช้ Python 3.11 ที่ลง PyTorch (CPU) ไว้แล้วเพื่อความเบา (บน Cloud Run มักใช้ CPU)
# ถ้าจะใช้ GPU บน GCE ต้องเปลี่ยน Base Image เป็น nvidia/cuda
FROM python:3.11-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Install System Dependencies (สำหรับ OpenCV Headless)
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libgl1 && rm -rf /var/lib/apt/lists/*

# 4. Copy Requirements & Install
COPY requirements.txt .
# แก้ไข requirements.txt ให้ใช้ torch cpu เพื่อลดขนาด image (ถ้า deploy บน Cloud Run)
# หรือใช้ requirements เดิมถ้า deploy บน VM ที่มี GPU
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --extra-index-url https://pypi.org/simple

# 5. Copy Code & Models
COPY src/ ./src/
COPY ocr_minimal/ ./ocr_minimal/
# COPY province_best.pth . (ถ้าจำเป็น)

# 6. Expose Port
EXPOSE 8080

# 7. Command to Run API
CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8080"]