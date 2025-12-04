# src/api_server.py
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import json
from pathlib import Path
import torchvision.transforms as T
from inference_sdk import InferenceHTTPClient

# Import ของเรา
from models import ResNetCRNN, ProvinceClassifier
from utils import beam_search_decode

app = FastAPI()

# --- LOAD MODELS (โหลดครั้งเดียวตอนเริ่ม Server) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Roboflow Client
RF_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="YOUR_ROBOFLOW_API_KEY" # ⚠️ ใส่ Key ของคุณ
)

# 2. Local Models
ocr_model = None
prov_model = None
int_to_char = {}

@app.on_event("startup")
async def load_models():
    global ocr_model, prov_model, int_to_char
    
    print("Loading models...")
    # Load Char Map
    with open("ocr_minimal/int_to_char.json", 'r', encoding='utf-8') as f:
        int_to_char = json.load(f)
    
    # Load OCR
    ocr_model = ResNetCRNN(1, len(int_to_char), hidden_size=256).to(DEVICE)
    ocr_model.load_state_dict(torch.load("ocr_minimal/best_model.pth", map_location=DEVICE)['model_state_dict'])
    ocr_model.eval()
    
    # Load Province
    prov_ckpt = torch.load("ocr_minimal/province_best.pth", map_location=DEVICE)
    prov_map = prov_ckpt['class_map']
    prov_model = ProvinceClassifier(len(prov_map)).to(DEVICE)
    prov_model.load_state_dict(prov_ckpt['model_state'])
    prov_model.eval()
    
    # Attach map to model for easy access
    prov_model.idx_to_prov = {int(k):v for k,v in prov_map.items()}
    print("Models loaded successfully!")

# Transforms
tf_ocr = T.Compose([T.Resize((64, 256)), T.ToTensor()])
tf_prov = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@app.post("/detect")
async def detect_license_plate(file: UploadFile = File(...)):
    # 1. Read Image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # 2. Detect Plate (Roboflow) - *ต้องระวังเรื่อง Latency*
    # ถ้าส่งรูปแบบ File ไป Roboflow ตรงๆ อาจจะช้า แนะนำให้ส่งเป็น Base64 หรือหาทาง Optimize
    # ในที่นี้ขอจำลองว่าส่งไป Roboflow แล้วได้ Box กลับมา
    # (ใน Production จริง อาจต้องใช้ Roboflow Docker Container เพื่อความเร็ว)
    
    # สมมติเรียก Roboflow แล้วได้ result (ต้องปรับโค้ดส่วนนี้ให้ส่งรูปไปได้จริง)
    # rf_result = RF_CLIENT.infer(...) 
    
    # เพื่อให้โค้ดรันได้ตอนนี้ ผมจะข้ามส่วน Roboflow Online ไปก่อน
    # แต่หลักการคือเอา Box มา Crop เหมือนใน full_inference.py
    
    results = []
    
    # ... (Logic การ Crop และ Predict เหมือน full_inference.py) ...
    # สมมติว่า Crop ได้ภาพ plate_crop และ prov_crop แล้ว
    
    # 3. OCR Inference
    plate_crop_gray = image.convert("L") # สมมติว่านี่คือภาพที่ครอปมาแล้ว
    tensor_ocr = tf_ocr(plate_crop_gray).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = ocr_model(tensor_ocr)
        text = beam_search_decode(out.log_softmax(-1), int_to_char)
        
    # 4. Province Inference
    tensor_prov = tf_prov(image).unsqueeze(0).to(DEVICE) # สมมติว่านี่คือภาพครอป
    with torch.no_grad():
        out = prov_model(tensor_prov)
        prov_name = prov_model.idx_to_prov.get(out.argmax(1).item(), "Unknown")
        
    return {
        "plate": text,
        "province": prov_name,
        "confidence": 0.95
    }

# วิธีรัน Server: uvicorn src.api_server:app --reload