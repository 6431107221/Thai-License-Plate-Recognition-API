# 🇹🇭 Thai License Plate Recognition API (End-to-End MLOps)
 
**สถานะ:** Production Ready (Deployed on GCP Cloud Run)
 
## ภาพรวมโครงการ (Project Overview)
 
โครงการนี้คือการพัฒนา API สำหรับระบบรู้จำป้ายทะเบียนรถยนต์ไทยแบบครบวงจร (End-to-End OCR Pipeline) โดยเน้นที่การสร้างสถาปัตยกรรมที่สามารถนำไปใช้งานจริง (Production-Grade) ด้วยแนวคิด **MLOps**
 
ระบบใช้โมเดล Deep Learning ขั้นสูง 2 ตัวทำงานร่วมกัน เพื่อให้ได้ความแม่นยำสูงทั้งการตรวจจับตำแหน่งและการรู้จำตัวอักษร
 
### จุดเด่นทางเทคนิคที่สำคัญ (Key Technical Highlights)
 
* **Two-Tier Detection:** ใช้ **Roboflow 3.0 Object Detection** ในการหาป้ายทะเบียน และใช้ **RF-DETR (Medium)** ในการแยกตัวอักษรและจังหวัดเพื่อ Segmentation
* **Custom Deep Learning:** พัฒนาโมเดล **CRNN** คู่กับ **CTC Loss** สำหรับการรู้จำตัวอักษรภาษาไทยโดยเฉพาะ
* **Containerization & Deployment:** ใช้ **Docker** และ Deploy เป็น Serverless Microservice บน **Google Cloud Run (GCP)**
 
---
 
## Technical Stack
 
| Category | Tools & Frameworks |
| :--- | :--- |
| **Deep Learning** | **PyTorch**, **CRNN** (Recognition), **CTC Loss**, Roboflow Inference SDK |
| **API / MLOps** | **Docker**, **FastAPI**, **Google Cloud Platform (GCP)**, Pipelining |
| **Language** | Python (Advanced), OpenCV |
 
---
 
## สถาปัตยกรรม (Hybrid 4-Stage Pipeline)
 
ระบบถูกออกแบบให้ทำงานเป็น Pipeline แบบ Multi-stage เพื่อเพิ่มความแม่นยำ:
 
1.  **Stage 1: Initial Detection:** ใช้ **Roboflow 3.0 Object Detection** ระบุตำแหน่งของแผ่นป้ายทะเบียนในภาพ Raw Image
2.  **Stage 2: Text Segmentation:** ใช้ **RF-DETR (Medium)** ครอบแยกตัวอักษร/เลขทะเบียน และจังหวัดจากป้ายที่ถูกตรวจจับมา
3.  **Stage 3: Recognition (OCR):** นำภาพที่ถูกแยกแล้วเข้าสู่โมเดล **CRNN** (ฝึกฝนด้วย **CTC Loss**) แปลงภาพเป็นข้อความ
4.  **Stage 4: Classification:** จำแนกหมวดหมู่จังหวัด (Optional Stage)
 
---
 
## การใช้งานและการ Deploy (MLOps Workflow)
 
### 1. การ Build Docker Image
 
ทำการ Build Docker Image โดยใช้ `Dockerfile` ที่อยู่ใน Root Directory:
 
```
docker build -t asia-southeast1-docker.pkg.dev/[PROJECT_ID]/ocr-api-repo/ocr-api-image:latest .
```
### 2. การ Deploy ไปยัง Google Cloud Run (GCP)

หลังจากยืนยันสิทธิ์ gcloud auth configure-docker แล้ว สามารถ Deploy Image ไปยัง GCP Cloud Run ได้ทันที
```
gcloud run deploy ocr-api-service \
  --image asia-southeast1-docker.pkg.dev/[PROJECT_ID]/ocr-api-repo/ocr-api-image:latest \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 512Mi \
  --max-instances 1
```
