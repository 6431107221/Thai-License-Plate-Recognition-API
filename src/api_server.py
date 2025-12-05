# src/api_server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import torch
import json
import cv2
import numpy as np
from pathlib import Path
import torchvision.transforms as T
from inference_sdk import InferenceHTTPClient

# Import Local Modules
from src.models import ResNetCRNN, ProvinceClassifier
from src.utils import beam_search_decode

app = FastAPI()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIG MODELS ---
# Roboflow Models
RF_API_KEY = "8Jx0yKiJpT5lb9rBGVzm"
MODEL_1_ID = "car-plate-detection-ahcak/3"  # หาป้ายทะเบียน
MODEL_2_ID = "ocr_prepare_test-tfc9g/4"     # หาตัวอักษร/จังหวัด ในป้าย

# Local Models
OCR_PATH = Path("ocr_minimal/best_model.pth")
PROV_PATH = Path("ocr_minimal/province_best.pth")
CHAR_MAP = Path("ocr_minimal/int_to_char.json")

# Global Vars
rf_client = None
ocr_model = None
prov_model = None
int_to_char = {}
prov_idx2prov = {}

# Transforms
tf_ocr = T.Compose([T.Resize((64, 256)), T.ToTensor()])
tf_prov = T.Compose([T.Resize((224, 224)), T.ToTensor(), 
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@app.on_event("startup")
async def startup_event():
    global rf_client, ocr_model, prov_model, int_to_char, prov_idx2prov
    print("Server Starting... Loading Models...")
    
    # 1. Setup Roboflow
    rf_client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=RF_API_KEY)
    
    # 2. Load Char Map
    with open(CHAR_MAP, 'r', encoding='utf-8') as f:
        int_to_char = json.load(f)
        
    # 3. Load OCR Model
    ocr_model = ResNetCRNN(1, len(int_to_char), hidden_size=256).to(DEVICE)
    ocr_ckpt = torch.load(OCR_PATH, map_location=DEVICE)
    ocr_model.load_state_dict(ocr_ckpt['model_state_dict'])
    ocr_model.eval()
    
    # 4. Load Province Model
    prov_ckpt = torch.load(PROV_PATH, map_location=DEVICE)
    prov_map = prov_ckpt['class_map']
    prov_idx2prov = {int(v): k for k, v in prov_map.items()} # เช็คทิศทาง Map อีกที
    # หมายเหตุ: ต้องตรวจสอบว่า model_state มีคำว่า model. นำหน้าหรือไม่ ถ้ามีต้องแก้ key เหมือนตอน inference.py
    # เพื่อความกระชับ ขอสมมติว่าโหลดได้ (ถ้าไม่ได้ ให้ใส่ Loop แก้ Key ตรงนี้)
    
    prov_model = ProvinceClassifier(len(prov_idx2prov)).to(DEVICE)
    
    # Auto-fix state dict keys
    state_dict = prov_ckpt['model_state']
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("model."): new_state_dict[f"model.{k}"] = v
        else: new_state_dict[k] = v
        
    prov_model.load_state_dict(new_state_dict)
    prov_model.eval()
    
    print("All Models Ready!")

@app.post("/detect")
async def detect_pipeline(file: UploadFile = File(...)):
    # 1. Read Raw Image
    image_data = await file.read()
    raw_img = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Save temp for Roboflow SDK (SDK ชอบ path file มากกว่า bytes)
    raw_img.save("temp_raw.jpg")
    
    # ==========================================
    # STEP 1: Detect Plate (Model 1)
    # ==========================================
    res_plate = rf_client.infer("temp_raw.jpg", model_id=MODEL_1_ID)
    
    # หา Box ที่มั่นใจที่สุดว่าเป็นป้ายทะเบียน
    plate_box = None
    max_conf = 0
    for pred in res_plate['predictions']:
        if pred['confidence'] > max_conf:
            max_conf = pred['confidence']
            # Convert Center-XYWH to Corner-XYXY
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            plate_box = (
                int(x - w/2), int(y - h/2), 
                int(x + w/2), int(y + h/2)
            )
            
    if not plate_box:
        return {"status": "failed", "message": "No license plate found"}
        
    # ==========================================
    # STEP 2: Crop Plate
    # ==========================================
    # Clamp coordinates (กันล้นขอบ)
    W, H = raw_img.size
    x1, y1, x2, y2 = plate_box
    plate_img = raw_img.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
    plate_img.save("temp_plate.jpg") # Save for Model 2

    # ==========================================
    # STEP 3: Detect License & Province (Model 2)
    # ==========================================
    res_components = rf_client.infer("temp_plate.jpg", model_id=MODEL_2_ID)
    
    license_crop = None
    province_crop = None
    
    # Loop หา component ภายในป้าย
    # คาดหวัง Class name: "license", "province" (เช็คใน Roboflow อีกที)
    for pred in res_components['predictions']:
        cls = pred['class']
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        box = (int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2))
        
        # Crop ย่อยจากภาพ Plate
        pW, pH = plate_img.size
        component_img = plate_img.crop((max(0, box[0]), max(0, box[1]), min(pW, box[2]), min(pH, box[3])))
        
        if "license" in cls or "text" in cls: # แก้ชื่อ Class ตามจริง
            license_crop = component_img
        elif "province" in cls: # แก้ชื่อ Class ตามจริง
            province_crop = component_img

    # ==========================================
    # STEP 4: Recognition (Local Models)
    # ==========================================
    result = {"plate": "", "province": ""}
    
    # 4.1 Recognize Characters
    if license_crop:
        gray = license_crop.convert("L")
        ts = tf_ocr(gray).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = ocr_model(ts)
            text = beam_search_decode(out.log_softmax(-1), int_to_char)
        result["plate"] = text
        
    # 4.2 Recognize Province
    if province_crop:
        rgb = province_crop.convert("RGB")
        ts = tf_prov(rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = prov_model(ts)
            idx = out.argmax(1).item()
            prov_name = prov_idx2prov.get(idx, str(idx))
        result["province"] = prov_name
    else:
        # Fallback: ถ้า Model 2 หาจังหวัดไม่เจอ อาจจะใช้ Heuristic Crop เหมือนเดิมก็ได้
        result["province"] = "Not Detected"

    return result