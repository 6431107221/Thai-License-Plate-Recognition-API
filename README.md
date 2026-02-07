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

## Result Screenshot
| Detection | Segmentation | Recognition |
|:---------:|:---------:|:------------------:|
| ![val_batch2_pred](https://github.com/user-attachments/assets/74c61b4c-eb9b-498e-bde5-82208c755cb5) | ![val_batch1_pred](https://github.com/user-attachments/assets/25f84e37-ec9d-4b36-9509-ac83e88542de) | ![WorKflow-LPR](https://github.com/user-attachments/assets/8cdd4102-4634-4643-9d15-3ad4c0c2c7a0) |

## Docker & GCP Deployment
```bash
gcloud projects create [PROJECT_NAME]
gcloud config set project [PROJECT_NAME]
gcloud services enable artifactregistry.googleapis.com
gcloud services enable run.googleapis.com
gcloud artifacts repositories create [REPO_NAME] [CONFIG_REPO]
gcloud auth configure-docker asia-southeast1-docker.pkg.dev
docker build -t asia-southeast1-docker.pkg.dev/[PROJECT_ID]/[REPO_NAME]/[IMAGE_NAME] .
docker tag [OLD_TAG] [NEW_TAG]
docker push [TAG]
gcloud run deploy [IMAGE_NAME]
  --image [image] `
  --region asia-southeast1 `
  --platform managed `
  --allow-unauthenticated `
  --memory 2Gi `
  --max-instances 1 `
  --min-instances 0 `
  --cpu-throttling
```
