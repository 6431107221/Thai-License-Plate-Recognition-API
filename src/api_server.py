# src/api_server.py
from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageOps
import io
import torch
import json
import numpy as np
import time
import os

from inference_sdk import InferenceHTTPClient

# Import Config และ Modules
# ต้องแน่ใจว่า structure ไฟล์ของคุณคือ src.config นะครับ
from src.config import cfg
from src.models import ResNetCRNN, ProvinceClassifier
from src.utils import beam_search_decode
from src.datasets import get_ocr_transforms, get_prov_transforms

app = FastAPI()

class LicensePlateService:
    def __init__(self):
        print("🚀 Initializing Service...")
        self.device = cfg.DEVICE
        
        # 1. Initialize Roboflow
        self.rf_client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com", 
            api_key=cfg.RF_API_KEY
        )
        
        # 2. Load Char Map
        self.int_to_char = {}
        if cfg.CHAR_MAP_PATH.exists():
            with open(cfg.CHAR_MAP_PATH, 'r', encoding='utf-8') as f:
                self.int_to_char = json.load(f)
        else:
            print(f"❌ Critical Error: Char map not found at {cfg.CHAR_MAP_PATH}")

        # 3. Load Models
        self.ocr_model = self._load_ocr_model()
        self.prov_model, self.prov_idx2prov = self._load_prov_model()
        
        # 4. Load Transforms
        self.tf_ocr = get_ocr_transforms(is_train=False)
        self.tf_prov = get_prov_transforms(is_train=False)

        print("✅ Service Ready!")

    def _load_ocr_model(self):
        path = cfg.OCR_MODEL_SAVE_PATH if cfg.OCR_MODEL_SAVE_PATH.exists() else cfg.OCR_PRETRAINED_PATH
        print(f"📂 Loading OCR Model from: {path}")
        
        model = ResNetCRNN(1, len(self.int_to_char), hidden_size=256).to(self.device)
        
        if path.exists():
            try:
                ckpt = torch.load(path, map_location=self.device, weights_only=True)
                state = ckpt.get('model_state_dict', ckpt)
                model.load_state_dict(state)
                model.eval()
                print("   ✅ OCR Loaded Successfully")
            except Exception as e:
                print(f"   ❌ OCR Load Failed: {e}")
        else:
             print("   ⚠️ No OCR model file found! Using random weights.")
        return model

    def _load_prov_model(self):
        path = cfg.PROV_MODEL_SAVE_PATH if cfg.PROV_MODEL_SAVE_PATH.exists() else cfg.PROV_PRETRAINED_PATH
        print(f"📂 Loading Province Model from: {path}")
        
        idx2prov = {}
        model = ProvinceClassifier(77).to(self.device)
        
        if path.exists():
            try:
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                
                if "class_map" in ckpt:
                    idx2prov = {int(k): v for k, v in ckpt["class_map"].items()}
                    model = ProvinceClassifier(len(idx2prov)).to(self.device)
                
                state = ckpt.get('model_state', ckpt)
                new_state = {f"model.{k}" if not k.startswith("model.") else k: v for k, v in state.items()}
                
                model.load_state_dict(new_state, strict=False)
                model.eval()
                print("   ✅ Province Loaded Successfully")
            except Exception as e:
                print(f"   ❌ Province Load Failed: {e}")
        else:
             print("   ⚠️ No Province model file found!")
        return model, idx2prov

    # ========================================================
    # 🎨 Preprocess: กลับไปใช้แบบดั้งเดิม (Simple Grayscale)
    # ========================================================
    def preprocess_image_simple(self, pil_img):
        # 1. Convert to Grayscale
        gray = pil_img.convert("L")
        
        # 2. Auto Contrast (แบบเดิมที่คุณเคยใช้แล้วเวิร์ค)
        # ช่วยดึงสีตัวอักษรให้เข้มขึ้นจากพื้นหลัง โดยไม่ทำลายรายละเอียดเหมือน Threshold
        gray = ImageOps.autocontrast(gray)
        
        return gray

    async def predict(self, image_bytes):
        req_id = int(time.time())
        
        # 1. Read Image
        raw_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if cfg.DEBUG_MODE: raw_img.save(cfg.DEBUG_IMAGE_DIR / f"{req_id}_0_raw.jpg")

        # 2. RF Detect Plate (Model 1)
        temp_path = cfg.DEBUG_IMAGE_DIR / "temp_rf.jpg"
        raw_img.save(temp_path)
        
        try:
            res_plate = self.rf_client.infer(str(temp_path), model_id=cfg.MODEL_DETECTION_ID)
            preds = res_plate.get('predictions', [])
        except Exception as e:
            if temp_path.exists(): os.remove(temp_path)
            return {"error": f"Model 1 Error: {e}"}

        if not preds:
            if temp_path.exists(): os.remove(temp_path)
            return {"status": "no_plate_found"}
        
        best_plate = max(preds, key=lambda x: x['confidence'])
        x, y, w, h = best_plate['x'], best_plate['y'], best_plate['width'], best_plate['height']
        
        W, H = raw_img.size
        margin = 0.05 
        x1 = max(0, int(x - w/2 - w*margin))
        y1 = max(0, int(y - h/2 - h*margin))
        x2 = min(W, int(x + w/2 + w*margin))
        y2 = min(H, int(y + h/2 + h*margin))
        
        plate_crop = raw_img.crop((x1, y1, x2, y2))
        if cfg.DEBUG_MODE: plate_crop.save(cfg.DEBUG_IMAGE_DIR / f"{req_id}_1_plate.jpg")

        # 3. Detect Components (Model 2)
        plate_crop.save(temp_path)
        try:
            res_comp = self.rf_client.infer(str(temp_path), model_id=cfg.MODEL_OCR_PREP_ID)
            preds_comp = res_comp.get('predictions', [])
        except:
            preds_comp = []
        
        if temp_path.exists(): os.remove(temp_path)

        license_crop = None
        province_crop = None
        pW, pH = plate_crop.size
        
        for pred in preds_comp:
            cls = pred['class']
            px, py, pw, ph_box = pred['x'], pred['y'], pred['width'], pred['height']
            
            bx1 = max(0, int(px - pw/2))
            by1 = max(0, int(py - ph_box/2))
            bx2 = min(pW, int(px + pw/2))
            by2 = min(pH, int(py + ph_box/2))
            
            comp_img = plate_crop.crop((bx1, by1, bx2, by2))
            
            if "Plate" in cls:
                license_crop = comp_img
            elif "Province" in cls:
                province_crop = comp_img

        # Fallback
        if not license_crop:
            license_crop = plate_crop.crop((0, 0, pW, int(pH*0.65)))
        if not province_crop:
            province_crop = plate_crop.crop((0, int(pH*0.6), pW, pH))

        # 4. Preprocess (Simple Grayscale)
        ocr_input = self.preprocess_image_simple(license_crop)
        prov_input = self.preprocess_image_simple(province_crop)

        if cfg.DEBUG_MODE:
            ocr_input.save(cfg.DEBUG_IMAGE_DIR / f"{req_id}_2_ocr_input.jpg")
            prov_input.save(cfg.DEBUG_IMAGE_DIR / f"{req_id}_3_prov_input.jpg")

        # 5. Inference
        result = {
            "RequestID": req_id,
            "Plate": "", 
            "Province": "",
            "Conf_Prov": 0.0
        }

        try:
            ts = self.tf_ocr(ocr_input).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.ocr_model(ts)
                result["Plate"] = beam_search_decode(out[0].log_softmax(-1), self.int_to_char)
        except Exception as e:
            print(f"OCR Error: {e}")

        try:
            ts = self.tf_prov(prov_input).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.prov_model(ts)
                probs = torch.softmax(out, dim=1)
                conf, idx = probs.max(1)
                result["Province"] = self.prov_idx2prov.get(idx.item(), str(idx.item()))
                result["Conf_Prov"] = float(conf.item())
        except Exception as e:
            print(f"Prov Error: {e}")

        return result

# --- Init ---
service = None

@app.on_event("startup")
async def startup_event():
    global service
    service = LicensePlateService()

@app.post("/detect")
async def detect_endpoint(file: UploadFile = File(...)):
    image_data = await file.read()
    return await service.predict(image_data)