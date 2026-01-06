# Thai License Plate Recognition API

A high-performance microservice for Thai license plate detection and recognition. This system utilizes a multi-stage deep learning pipeline to extract plate numbers and province names from images.

## Features
- Plate Detection and Segmentation using YOLO11.
- Custom OCR (CRNN) specialized for Thai characters.
- Province Classification using MobileNetV2.
- Ready-to-deploy FastAPI backend.
- Dockerized for cloud deployment.

## Tech Stack
- AI/ML: PyTorch, Ultralytics (YOLO11), OpenCV.
- Backend: FastAPI, Uvicorn.
- Infrastructure: Docker, GCP Cloud Run.

## Architecture
1. Detection: Locate the license plate in the image.
2. Segmentation: Separate the plate number and province area.
3. Recognition: OCR for characters and Classification for province.

![val_batch2_pred](https://github.com/user-attachments/assets/74c61b4c-eb9b-498e-bde5-82208c755cb5)

![val_batch1_pred](https://github.com/user-attachments/assets/25f84e37-ec9d-4b36-9509-ac83e88542de)

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API:
   ```bash
   uvicorn src.api_server:app --reload
   ```

## Docker Deployment
```bash
docker build -t thai-lpr-api .
docker run -p 8080:8080 thai-lpr-api
```

---
Developed for professional Thai license plate recognition tasks.
