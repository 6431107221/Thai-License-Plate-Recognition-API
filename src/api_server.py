
# src/api_server.py
from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageOps
import io
import torch
import json
import numpy as np
import time
import os
from ultralytics import YOLO
import torch.nn.functional as F

# Import Config และ Modules
from src.config import cfg
from src.models import ResNetCRNN, ProvinceClassifier, best_path_decode
from src.preprocess import preprocess_raw_image, get_ocr_transforms, get_prov_transforms

import shutil

app = FastAPI(title="Thai LPR API Service")

# Setup Debug Directory
DEBUG_API_DIR = cfg.DEBUG_IMAGE_DIR

class LicensePlateService:
    def __init__(self):
        print("Initializing Local LPR Service (Synced with latest models)...")
        self.device = cfg.DEVICE
        
        # Ensure debug directory exists
        if DEBUG_API_DIR.exists():
            shutil.rmtree(DEBUG_API_DIR)
        DEBUG_API_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. Load Local YOLO Models
        try:
            self.model_plate = YOLO(cfg.MODEL_DETECTION_PATH)
            self.model_comp  = YOLO(cfg.MODEL_OCR_PREP_PATH)
            print("   YOLO Models Loaded")
        except Exception as e:
            print(f"   Error loading YOLO: {e}")

        # 2. Load Engine Models (OCR & Province) with Embedded Maps
        self.ocr_model, self.int_to_char = self._load_ocr_engine()
        self.prov_model, self.i2p = self._load_prov_engine()
        
        # 3. Transforms
        self.tf_ocr = get_ocr_transforms(is_train=False)
        self.tf_prov = get_prov_transforms(is_train=False)

        print("Service is UP and READY!")

    def _load_ocr_engine(self):
        print("   Loading OCR Engine...")
        if not cfg.OCR_MODEL_SAVE_PATH.exists():
            print(f"   OCR model not found: {cfg.OCR_MODEL_SAVE_PATH}")
            return None, {}
            
        ckpt = torch.load(cfg.OCR_MODEL_SAVE_PATH, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        int_to_char = ckpt.get("int_to_char", {})
        
        model = ResNetCRNN(img_channel=1, num_classes=len(int_to_char)).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model, int_to_char

    def _load_prov_engine(self):
        print("   Loading Province Engine...")
        if not cfg.PROV_MODEL_SAVE_PATH.exists():
            print(f"   Province model not found: {cfg.PROV_MODEL_SAVE_PATH}")
            return None, {}
            
        ckpt = torch.load(cfg.PROV_MODEL_SAVE_PATH, map_location=self.device)
        class_map = ckpt.get("class_map", {})
        i2p = {int(k) if str(k).isdigit() else k: v for k, v in class_map.items()}
        state_dict = ckpt.get("model_state", ckpt.get("model", ckpt))
        
        model = ProvinceClassifier(n_classes=len(class_map)).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model, i2p

    async def predict(self, image_bytes):
        # Clear debug dir for new request
        if DEBUG_API_DIR.exists():
            for f in DEBUG_API_DIR.glob("*"):
                if f.is_file(): f.unlink()
        
        start_time = time.time()
        raw_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        raw_img.save(DEBUG_API_DIR / "0_input_raw.jpg")

        # 1. YOLO Plate Detect
        results_plate = self.model_plate(raw_img, verbose=False)[0]
        if len(results_plate.boxes) == 0:
            return {"status": "error", "message": "No license plate detected"}

        detections = []
        for i, box in enumerate(results_plate.boxes):
            conf_plate = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            plate_crop = raw_img.crop((x1, y1, x2, y2))
            plate_crop.save(DEBUG_API_DIR / f"1_plate_crop_{i}.jpg")

            # 2. YOLO Component
            results_comp = self.model_comp(plate_crop, verbose=False)[0]
            
            ocr_crop = None
            prov_crop = None

            for j, c_box in enumerate(results_comp.boxes):
                cls_idx = int(c_box.cls[0])
                cls_name = self.model_comp.names[cls_idx]
                bx1, by1, bx2, by2 = c_box.xyxy[0].cpu().numpy().astype(int)
                comp_img = plate_crop.crop((bx1, by1, bx2, by2))
                
                if "Plate" in cls_name: 
                    ocr_crop = comp_img
                    ocr_crop.save(DEBUG_API_DIR / f"2_ocr_target_{i}.jpg")
                elif "Province" in cls_name: 
                    prov_crop = comp_img
                    prov_crop.save(DEBUG_API_DIR / f"3_prov_target_{i}.jpg")

            # Fallback crops logic (if YOLO components fail)
            pW, pH = plate_crop.size
            if not ocr_crop: 
                ocr_crop = plate_crop.crop((0, 0, pW, int(pH*0.65)))
                
            if not prov_crop: 
                prov_crop = plate_crop.crop((0, int(pH*0.6), pW, pH))
                prov_crop.save(DEBUG_API_DIR / f"3_prov_target_{i}_fallback.jpg")

            # PREPROCESSING ENHANCEMENT
            ocr_crop = ImageOps.autocontrast(ocr_crop, cutoff=1)
            ocr_crop.save(DEBUG_API_DIR / f"2_ocr_target_{i}_enhanced.jpg")

            # 3. Engines Inference
            plate_text = ""
            province_name = "Unknown"
            conf_prov = 0.0

            # OCR Inference
            ts_ocr = self.tf_ocr(ocr_crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out_ocr = self.ocr_model(ts_ocr)
                plate_text = best_path_decode(out_ocr.softmax(-1), self.int_to_char)[0]

            # Province Inference
            ts_prov = self.tf_prov(prov_crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out_prov = self.prov_model(ts_prov)
                probs = F.softmax(out_prov, dim=1)
                p_conf, p_idx = probs.max(1)
                province_name = self.i2p.get(p_idx.item(), "Unknown")
                conf_prov = float(p_conf.item())

            detections.append({
                "plate_text": plate_text,
                "province": province_name,
                "confidence": {
                    "plate_detection": conf_plate,
                    "province": conf_prov
                },
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })

        return {
            "status": "success",
            "results": detections,
            "latency_ms": int((time.time() - start_time) * 1000)
        }

# --- Shared instances ---
service = None

@app.on_event("startup")
async def startup_event():
    global service
    service = LicensePlateService()

@app.post("/detect")
async def detect_endpoint(file: UploadFile = File(...)):
    image_data = await file.read()
    return await service.predict(image_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)