# src/api_server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, ImageOps
import io
import torch
import json
import cv2
import numpy as np
from pathlib import Path
import torchvision.transforms as T
from inference_sdk import InferenceHTTPClient
import time
import os

# Import Local Modules
from src.models import ResNetCRNN, ProvinceClassifier
from src.utils import beam_search_decode

#  IMPORT DATASET (แก้ตรงนี้: เรียกใช้ Transform ที่ถูกต้อง) 
from src.datasets import get_ocr_transforms, get_prov_transforms

app = FastAPI()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIG ---
RF_API_KEY = "8Jx0yKiJpT5lb9rBGVzm"
MODEL_1_ID = "car-plate-detection-ahcak/3"
MODEL_2_ID = "ocr_prepare_test-tfc9g/4"

OCR_PATH = Path("ocr_train_out/best_model.pth")
PROV_PATH = Path("ocr_train_out/province_best.pth")
CHAR_MAP = Path("ocr_minimal/int_to_char.json")

# Debug Folder
DEBUG_DIR = Path("debug_images")
DEBUG_DIR.mkdir(exist_ok=True)

# Global Vars
rf_client = None
ocr_model = None
prov_model = None
int_to_char = {}
prov_idx2prov = {}

#  FIX: ใช้ Transform มาตรฐานเดียวกับ Dataset (สำคัญมาก!) 
tf_ocr = get_ocr_transforms(is_train=False)
tf_prov = get_prov_transforms(is_train=False)

@app.on_event("startup")
async def startup_event():
    global rf_client, ocr_model, prov_model, int_to_char, prov_idx2prov
    print(" Server Starting... Loading Models...")
    
    rf_client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=RF_API_KEY)
    
    if CHAR_MAP.exists():
        with open(CHAR_MAP, 'r', encoding='utf-8') as f:
            int_to_char = json.load(f)
            
    if OCR_PATH.exists():
        ocr_model = ResNetCRNN(1, len(int_to_char), hidden_size=256).to(DEVICE)
        try:
            ckpt = torch.load(OCR_PATH, map_location=DEVICE, weights_only=True)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            ocr_model.load_state_dict(state)
            ocr_model.eval()
            print(" OCR Model Loaded")
        except: pass

    if PROV_PATH.exists():
        try:
            ckpt = torch.load(PROV_PATH, map_location=DEVICE, weights_only=True)
            if 'class_map' in ckpt:
                prov_map = ckpt['class_map']
                prov_idx2prov = {int(k): v for k, v in prov_map.items()}
                
                prov_model = ProvinceClassifier(len(prov_idx2prov)).to(DEVICE)
                state_dict = ckpt['model_state']
                new_state_dict = {}
                for k, v in state_dict.items():
                    if not k.startswith("model."): new_state_dict[f"model.{k}"] = v
                    else: new_state_dict[k] = v
                prov_model.load_state_dict(new_state_dict)
                prov_model.eval()
                print(" Province Model Loaded")
        except: pass
        
    print(" All Models Ready!")

@app.post("/detect")
async def detect_pipeline(file: UploadFile = File(...)):
    req_id = int(time.time())
    print(f"\n--- Processing Request ID: {req_id} ---")

    # 1. Read Image
    image_data = await file.read()
    raw_img_rgb = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Save Raw for Debug
    raw_img_rgb.save(DEBUG_DIR / f"{req_id}_0_raw.jpg")
    
    # Prepare Grayscale for Roboflow
    raw_img_gray = raw_img_rgb.convert("L")
    temp_raw_path = DEBUG_DIR / f"{req_id}_temp_raw_gray.jpg"
    raw_img_gray.save(temp_raw_path)
    
    # ==========================================
    # STEP 1: Detect Plate (Model 1)
    # ==========================================
    try:
        res_plate = rf_client.infer(str(temp_raw_path), model_id=MODEL_1_ID)
    except Exception as e:
        return {"error": f"Model 1 Failed: {e}"}

    plate_box = None
    max_conf = 0
    for pred in res_plate['predictions']:
        if pred['confidence'] > max_conf:
            max_conf = pred['confidence']
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            plate_box = (int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2))
            
    if not plate_box:
        return {"status": "failed", "message": "No license plate found"}
        
    # ==========================================
    # STEP 2: Crop Plate
    # ==========================================
    W, H = raw_img_rgb.size
    x1, y1, x2, y2 = plate_box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    
    plate_img_rgb = raw_img_rgb.crop((x1, y1, x2, y2))
    
    # Crop Gray (ส่งไป Model 2)
    plate_img_gray = plate_img_rgb.convert("L")
    temp_plate_path = DEBUG_DIR / f"{req_id}_temp_plate_gray.jpg"
    plate_img_gray.save(temp_plate_path)
    
    plate_img_rgb.save(DEBUG_DIR / f"{req_id}_1_crop_plate.jpg")

    # ==========================================
    # STEP 3: Detect Components (Model 2)
    # ==========================================
    try:
        res_components = rf_client.infer(str(temp_plate_path), model_id=MODEL_2_ID)
    except:
        res_components = {'predictions': []}
    
    license_crop = None
    province_crop = None
    
    pW, pH = plate_img_rgb.size
    
    for pred in res_components['predictions']:
        cls = pred['class']
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        box = (int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2))
        
        bx1, by1 = max(0, box[0]), max(0, box[1])
        bx2, by2 = min(pW, box[2]), min(pH, box[3])
        
        component_img = plate_img_rgb.crop((bx1, by1, bx2, by2))
        
        if "Plate" in cls:
            license_crop = component_img
        elif "Province" in cls:
            province_crop = component_img

    # ==========================================
    # STEP 4: Recognition
    # ==========================================
    result = {
        "RequestID": req_id,
        "Plate": "", 
        "Province": "", 
        "Conf_Prov": 0.0,
        "Method": ""
    }
    
    # --- 4.1 OCR ---
    img_for_ocr = license_crop if license_crop else plate_img_rgb
    img_for_ocr.save(DEBUG_DIR / f"{req_id}_2_input_ocr.jpg")
    
    if img_for_ocr:
        try:
            #  แปลงเป็น Grayscale ก่อนส่งเข้า Transform
            gray = img_for_ocr.convert("L")
            
            # SmartResize จะทำงานใน tf_ocr นี้
            ts = tf_ocr(gray).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out = ocr_model(ts)
                text = beam_search_decode(out[0].log_softmax(-1), int_to_char)
            result["Plate"] = text
        except Exception as e:
            print(f"OCR Error: {e}")

    # --- 4.2 Province ---
    img_for_prov = province_crop
    result["Method"] = "Roboflow"
    
    if not img_for_prov:
        img_for_prov = plate_img_rgb.crop((0, int(pH*0.65), pW, pH))
        result["Method"] = "Heuristic (Fallback)"

    img_for_prov.save(DEBUG_DIR / f"{req_id}_3_input_prov.jpg")

    if img_for_prov:
        try:
            #  ไม่ต้องแปลง Fake RGB เองแล้ว ปล่อยให้ Transform จัดการ
            # ส่งภาพเข้า tf_prov ได้เลย (มันจะเช็คและแปลงให้ถ้าจำเป็น)
            # แต่ปกติ SmartResize รองรับ PIL RGB/L อยู่แล้ว
            
            # ถ้า img_for_prov เป็น RGB ก็ส่ง RGB
            # ถ้าเป็น L ก็ส่ง L
            
            ts = tf_prov(img_for_prov).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out = prov_model(ts)
                probs = torch.softmax(out, dim=1)
                conf, idx = probs.max(1)
                prov_name = prov_idx2prov.get(idx.item(), str(idx.item()))
                
            result["Province"] = prov_name
            result["Conf_Prov"] = float(conf.item())
        except Exception as e:
            print(f"Province Error: {e}")

    if os.path.exists(temp_raw_path): os.remove(temp_raw_path)
    if os.path.exists(temp_plate_path): os.remove(temp_plate_path)

    return result