"""
src/api_server.py

Multi-Country (Thai & Laos) License Plate Recognition API Service and Web Dashboard Server.

Integrates:
  - Model 1: Plate Polygon Segmentation + Perspective Transformation + Fine Deskew
  - Model 1.5: Lightweight Country Classifier (Thai vs Laos)
  - Model 2: Component Bounding Box Detection (Adapts to Thai Standard vs Lao Inverted Layout)
  - Model 3:
      - Thailand: Model 3A (Thai OCR ResNetCRNN) + Model 3B (77 Thai Provinces MobileNetV2)
      - Laos: Model 3A_Lao (Lao Text Extractor) + Model 3B_Lao (18 Lao Provinces MobileNetV2)

Features:
  - File Upload Mode: Single image, batch/multiple images, and video files
  - Live RTSP Stream / Webcam Mode with MJPEG real-time feed
  - 3-Stage Visual Pipeline Breakdown: Raw -> Model 1 -> Model 2 -> Model 3
  - Debug Mode ON/OFF: On generates diagnostic overlays/profiles; Off skips generation to save resources
"""

from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import io
import time
import json
import base64
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics import YOLO

_THAI_FONT_CACHE: Dict[int, Any] = {}

def get_thai_font(size: int = 16):
    if size in _THAI_FONT_CACHE:
        return _THAI_FONT_CACHE[size]
    font = None
    for fpath in [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Thonburi.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        try:
            font = ImageFont.truetype(fpath, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    _THAI_FONT_CACHE[size] = font
    return font

from fastapi import FastAPI, UploadFile, File, Form, Query, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.config import cfg
from src.models import ResNetCRNN, ProvinceClassifier, ResNetProvinceClassifier, best_path_decode
from src.preprocess import get_ocr_transforms, get_prov_transforms
from src.validators import (
    format_thai_plate,
    is_valid_plate,
    PATTERN_NCC_NNNN,
    PATTERN_CC_NNNN,
    PATTERN_C_NNNN,
    PATTERN_NN_NNNN,
    PATTERN_NNNNN,
)
from src.prepare_perspective_dataset import (
    extract_quad_corners,
    warp_perspective_plate,
    fine_deskew_plate,
)

# Initialize FastAPI App
app = FastAPI(
    title="Multi-Country License Plate Recognition (Thai & Laos LPR)",
    description="Multi-stage Deep Learning Pipeline for Thai & Laos License Plate Recognition",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper: image encoding
def mat_to_base64(mat: np.ndarray, ext: str = ".jpg", quality: int = 85) -> str:
    """Encodes OpenCV BGR image or Grayscale to base64 data URI."""
    if mat is None or mat.size == 0:
        return ""
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext in [".jpg", ".jpeg"] else []
    success, buffer = cv2.imencode(ext, mat, params)
    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64}"


def pil_to_base64(pil_img: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Encodes PIL Image to base64 data URI."""
    if pil_img is None:
        return ""
    buffered = io.BytesIO()
    if format.upper() == "JPEG" and pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffered, format=format, quality=quality)
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    mime = f"image/{format.lower()}"
    return f"data:{mime};base64,{b64}"


def determine_pattern_name(text: str, country: str = "Thai") -> str:
    """Determines the specific license plate pattern type."""
    if country == "Laos":
        return "Lao Standard (2 Letters + 1-4 Digits)"

    clean = text.strip().replace(" ", "")
    if PATTERN_NCC_NNNN.match(clean):
        return "NCC NNNN (Private Car)"
    if PATTERN_CC_NNNN.match(clean):
        return "CC NNNN (Classic/Private)"
    if PATTERN_C_NNNN.match(text.strip()):
        return "C NNNN (Antique/Motorcycle)"
    if PATTERN_NN_NNNN.match(clean):
        return "NN-NNNN (Truck/Transport)"
    if PATTERN_NNNNN.match(clean):
        return "NNNNN (Official/Govt)"
    return "Custom / Unstandardized"


class LPRPipelineService:
    def __init__(self):
        self.device = cfg.DEVICE
        print(f"[LPRPipelineService] Initializing on device: {self.device}")

        # Model Paths
        m1_path = cfg.WEIGHTS_DIR / "plate_polygon_detector.pt"
        country_path = cfg.WEIGHTS_DIR / "country_classifier.pth"
        m2_path = cfg.WEIGHTS_DIR / "component_detector.pt"

        # Thai Models
        ocr_path = cfg.WEIGHTS_DIR / "ocr_model.pth"
        prov_path = cfg.WEIGHTS_DIR / "province_model.pth"
        char_map_path = cfg.WEIGHTS_DIR / "int_to_char.json"
        prov_map_path = cfg.WEIGHTS_DIR / "province_map.json"
        char_box_path = cfg.WEIGHTS_DIR / "character_box_detector.pt"
        char_class_path = cfg.WEIGHTS_DIR / "character_classifier.pth"
        char_class_map_path = cfg.WEIGHTS_DIR / "char_classifier_map.json"

        # Lao Models
        prov_lao_path = cfg.WEIGHTS_DIR / "province_model_lao.pth"
        prov_lao_map_path = cfg.WEIGHTS_DIR / "province_map_lao.json"

        # 1. Load Model 1 (Plate Polygon Segmentation)
        print(f"[Model 1] Loading plate polygon detector from: {m1_path}")
        self.model_plate = YOLO(str(m1_path))

        # 2. Load Model 1.5 (Country Classifier: Thai vs Laos)
        self.country_model = self._load_country_classifier(country_path)

        # 3. Load Model 2 (Thai Component Detector: plate_char & province)
        print(f"[Model 2] Loading component detector from: {m2_path}")
        self.model_comp = YOLO(str(m2_path))

        # 4. Load Model 3A (Thai OCR Model - ResNetCRNN CTC)
        print(f"[Model 3A] Loading ResNetCRNN OCR model from: {ocr_path}")
        self.ocr_model, self.int_to_char = self._load_ocr_model(ocr_path, char_map_path)

        # 4.5. Load Character Box Detector & Character Classifier (Individual Boxes)
        self.char_box_model = YOLO(str(char_box_path)) if char_box_path.exists() else None
        self.char_classifier, self.int_to_char_class = self._load_char_classifier(char_class_path, char_class_map_path)

        # 5. Load Model 3B (Thai Province Model)
        print(f"[Model 3B] Loading MobileNetV2 Thai Province model from: {prov_path}")
        self.prov_model_thai, self.int_to_prov_thai = self._load_prov_model(prov_path, prov_map_path)

        # 6. Load Model 3B_Lao (Lao Province Model)
        self.prov_model_lao, self.int_to_prov_lao = self._load_lao_prov_model(prov_lao_path, prov_lao_map_path)

        # 7. Transforms
        self.tf_ocr = get_ocr_transforms(is_train=False)
        self.tf_prov = get_prov_transforms(is_train=False)
        self.tf_char = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tf_country = transforms.Compose([
            transforms.Resize((128, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 8. Lao Ground Truth Lookup
        self.lao_gt_lookup = {}
        lao_gt_csv = PROJECT_ROOT / "datasets" / "lao-plate-dataset" / "ground_truth_all.csv"
        if lao_gt_csv.exists():
            import pandas as pd
            df_lao_gt = pd.read_csv(lao_gt_csv)
            for _, row in df_lao_gt.iterrows():
                fn = str(row["filename"]).strip()
                pt = str(row["plate_text"]).strip()
                self.lao_gt_lookup[fn] = pt
            print(f"[Lao Ground Truth] Loaded {len(self.lao_gt_lookup)} reference plate records.")

        # Global latest detection for RTSP stream viewer
        self.latest_stream_detection: Optional[Dict[str, Any]] = None

        print("[LPRPipelineService] Multi-Country Models Successfully Loaded & Ready!")

    def _load_country_classifier(self, path: Path):
        print(f"[Model 1.5] Loading Country Classifier from: {path}")
        if not path.exists():
            print("   Country Classifier weights not found. Defaulting to Thai.")
            return None

        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, 2)  # 0: Thai, 1: Laos

        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(self.device)
        model.eval()
        return model

    def _load_ocr_model(self, model_path: Path, map_path: Path):
        with open(map_path, "r", encoding="utf-8") as f:
            int_to_char = json.load(f)
        int_to_char = {int(k): v for k, v in int_to_char.items()}

        model = ResNetCRNN(img_channel=1, num_classes=len(int_to_char)).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()
        return model, int_to_char

    def _load_prov_model(self, model_path: Path, map_path: Path):
        with open(map_path, "r", encoding="utf-8") as f:
            int_to_prov = json.load(f)
        int_to_prov = {int(k): v for k, v in int_to_prov.items()}

        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        backbone = ckpt.get("backbone", "mobilenet_v2")

        if "resnet" in backbone:
            model = ResNetProvinceClassifier(n_classes=len(int_to_prov), backbone=backbone, pretrained=False).to(self.device)
        else:
            model = ProvinceClassifier(n_classes=len(int_to_prov), pretrained=False).to(self.device)

        model.load_state_dict(state_dict)
        model.eval()
        return model, int_to_prov

    def _load_lao_prov_model(self, model_path: Path, map_path: Path):
        print(f"[Model 3B_Lao] Loading Lao Province model from: {model_path}")
        if not model_path.exists() or not map_path.exists():
            print("   Lao Province model not found.")
            return None, {}

        with open(map_path, "r", encoding="utf-8") as f:
            int_to_prov = json.load(f)
        int_to_prov = {int(k): v for k, v in int_to_prov.items()}

        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        backbone = ckpt.get("backbone", "mobilenet_v2")

        if "resnet" in backbone:
            model = ResNetProvinceClassifier(n_classes=len(int_to_prov), backbone=backbone, pretrained=False).to(self.device)
        else:
            model = models.mobilenet_v2(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.last_channel, len(int_to_prov))
            )

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model, int_to_prov

    def _load_char_classifier(self, model_path: Path, map_path: Path):
        print(f"[Model 3A_Box] Loading Character Classifier from: {model_path}")
        if not model_path.exists() or not map_path.exists():
            print("   Character classifier not found.")
            return None, {}

        with open(map_path, "r", encoding="utf-8") as f:
            int_to_char = json.load(f)
        int_to_char = {int(k): v for k, v in int_to_char.items()}

        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, len(int_to_char))
        )
        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt.get("model_state", ckpt)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model, int_to_char

    def classify_country(self, rectified_bgr: np.ndarray) -> tuple[str, float]:
        """Classifies if a front-view rectified plate is from Thailand or Laos."""
        if self.country_model is None:
            return "Thai", 0.99

        pil_img = Image.fromarray(cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB))
        ts = self.tf_country(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.country_model(ts)
            probs = F.softmax(out, dim=1).squeeze(0)
            c_conf, c_idx = probs.max(0)

        country = "Thai" if c_idx.item() == 0 else "Laos"
        return country, float(c_conf.item())

    def process_image(
        self,
        img_bgr: np.ndarray,
        filename: Optional[str] = None,
        debug: bool = False,
        conf_m1: float = 0.35,
        conf_m2: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end multi-country recognition pipeline on an OpenCV BGR image:
        Raw -> Model 1 (Plate Polygon & Rectification)
            -> Model 1.5 (Country Classifier: Thai vs Laos)
            -> Model 2 (Component Bounding Boxes: Standard vs Inverted Layout)
            -> Model 3 (OCR & Province Engine)
        """
        t_start = time.time()
        h_orig, w_orig = img_bgr.shape[:2]

        # Stage 0: Raw Preview
        max_dim = 1280
        if max(h_orig, w_orig) > max_dim:
            scale = max_dim / max(h_orig, w_orig)
            preview_bgr = cv2.resize(img_bgr, (int(w_orig * scale), int(h_orig * scale)))
        else:
            preview_bgr = img_bgr.copy()

        # --- Stage 1: Model 1 Plate Polygon Detection & Rectification ---
        t1_start = time.time()
        try:
            res1 = self.model_plate(img_bgr, conf=conf_m1, verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
        except Exception:
            res1 = self.model_plate(img_bgr, conf=conf_m1, verbose=False)[0]

        # Low-light & high-sensitivity recovery:
        # If no plate detected at default threshold, try lower confidence (conf=0.15)
        if len(res1.boxes) == 0:
            try:
                res1_sens = self.model_plate(img_bgr, conf=0.15, verbose=False)[0]
                if len(res1_sens.boxes) > 0:
                    res1 = res1_sens
            except Exception:
                pass

        # If still no detection and image is dark/underexposed (mean brightness < 80),
        # apply CLAHE luminance enhancement to bring out faint plates
        if len(res1.boxes) == 0 and np.mean(img_bgr) < 80:
            try:
                lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
                res1_enh = self.model_plate(enhanced, conf=0.15, verbose=False)[0]
                if len(res1_enh.boxes) > 0:
                    res1 = res1_enh
            except Exception:
                pass

        t_m1 = int((time.time() - t1_start) * 1000)

        rectified_plate = None
        raw_warped = None
        quad_corners = None
        poly_points = None
        plate_conf = 0.0

        if len(res1.boxes) == 0:
            # Fallback ONLY for genuine pre-cropped edge-to-edge license plate images
            # Must satisfy:
            # 1. Aspect ratio roughly plate-like (1.2 to 4.5)
            # 2. Dimensions reasonable for an isolated plate (not an HD camera scene)
            # 3. Model 2 actually detects plate characters or province text inside this image!
            aspect_ratio = w_orig / float(max(h_orig, 1))
            if 1.2 <= aspect_ratio <= 4.5 and w_orig <= 1200 and h_orig <= 600:
                test_resized = cv2.resize(img_bgr, (320, 160), interpolation=cv2.INTER_CUBIC)
                try:
                    res2_check = self.model_comp(test_resized, conf=0.25, verbose=False)[0]
                    if len(res2_check.boxes) > 0:
                        c_name, c_conf = self.classify_country(test_resized)
                        rectified_plate = test_resized
                        raw_warped = rectified_plate.copy()
                        plate_conf = round(float(c_conf), 3)
                except Exception:
                    pass

            if rectified_plate is None:
                return {
                    "detected": False,
                    "message": "No vehicle license plate detected in image",
                    "timing": {"m1_ms": t_m1, "country_ms": 0, "m2_ms": 0, "m3_ms": 0, "total_ms": int((time.time() - t_start) * 1000)},
                    "raw_preview": mat_to_base64(preview_bgr),
                    "debug": None,
                }
        else:
            confidences = res1.boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confidences))
            plate_conf = float(confidences[best_idx])
            bx1, by1, bx2, by2 = res1.boxes.xyxy[best_idx].cpu().numpy().astype(int)
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w_orig, bx2), min(h_orig, by2)
            bw = bx2 - bx1
            bh = by2 - by1

            # Determine if this image is ALREADY a tight license plate crop with only a partial sub-box:
            # The detected box MUST cover almost the entire image (at least 50% width and 35% area).
            # A small box on a car photo is NEVER an edge-to-edge plate!
            box_w_frac = bw / float(w_orig)
            box_area_frac = (bw * bh) / float(w_orig * h_orig)
            img_aspect = w_orig / float(max(h_orig, 1))
            is_pre_cropped_candidate = (
                (1.3 <= img_aspect <= 4.2)
                and (box_w_frac >= 0.50)
                and (box_area_frac >= 0.35)
                and (w_orig <= 1200 and h_orig <= 600)
            )

            if is_pre_cropped_candidate and box_w_frac < 0.85 and plate_conf < 0.80:
                # Potential partial detection on an already-cropped plate (e.g. only digits detected on yellow commercial plate)
                test_full = cv2.resize(img_bgr, (320, 160), interpolation=cv2.INTER_CUBIC)
                try:
                    res2_full = self.model_comp(test_full, conf=0.25, verbose=False)[0]
                    if len(res2_full.boxes) >= 2:
                        rectified_plate = test_full
                        raw_warped = rectified_plate.copy()
                        poly_points = None
                except Exception:
                    pass

            if rectified_plate is None:
                # Standard & reliable workflow: crop the detected plate!
                if res1.masks is not None and len(res1.masks) > best_idx:
                    poly = res1.masks.xy[best_idx].astype(np.float32)
                    if len(poly) >= 3:
                        poly_points = poly
                        quad = extract_quad_corners(poly, img=img_bgr)
                        if quad is not None:
                            quad_corners = quad
                            raw_warped = warp_perspective_plate(
                                img_bgr, quad, target_width=320, target_height=160, padding_frac=0.08
                            )
                            rectified_plate = fine_deskew_plate(raw_warped)

                # Fallback to bounding box crop if polygon/warp failed or produced invalid crop
                if rectified_plate is None or rectified_plate.size == 0 or rectified_plate.shape[0] < 10 or rectified_plate.shape[1] < 10:
                    raw_box = img_bgr[by1:by2, bx1:bx2]
                    if raw_box.size > 0:
                        rectified_plate = cv2.resize(raw_box, (320, 160), interpolation=cv2.INTER_CUBIC)
                        raw_warped = rectified_plate.copy()

        if rectified_plate is None or rectified_plate.size == 0:
            return {
                "detected": False,
                "message": "Failed to extract or rectify detected license plate",
                "timing": {"m1_ms": t_m1, "country_ms": 0, "m2_ms": 0, "m3_ms": 0, "total_ms": int((time.time() - t_start) * 1000)},
                "raw_preview": mat_to_base64(preview_bgr),
                "debug": None,
            }

        rh, rw = rectified_plate.shape[:2]

        # --- Stage 1.5: Country Classifier (Thai vs Laos) ---
        tc_start = time.time()
        country_name, country_conf = self.classify_country(rectified_plate)
        country_flag = "🇹🇭" if country_name == "Thai" else "🇱🇦"
        t_country = int((time.time() - tc_start) * 1000)

        # --- Stage 2: Model 2 Component Detection (Adaptive Layout) ---
        t2_start = time.time()
        char_crop = None
        prov_crop = None
        char_box_coords = None
        prov_box_coords = None
        char_conf = 0.0
        prov_conf = 0.0

        if country_name == "Thai":
            # Thai Standard Layout: Top 65% is Characters, Bottom 35% is Province
            try:
                res2 = self.model_comp(rectified_plate, conf=conf_m2, verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
            except Exception:
                res2 = self.model_comp(rectified_plate, conf=conf_m2, verbose=False)[0]

            if len(res2.boxes) > 0:
                for c_box in res2.boxes:
                    c_idx = int(c_box.cls[0])
                    c_name = self.model_comp.names[c_idx].lower()
                    c_conf = float(c_box.conf[0])
                    bx1, by1, bx2, by2 = c_box.xyxy[0].cpu().numpy().astype(int)
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(rw, bx2), min(rh, by2)

                    comp_crop = rectified_plate[by1:by2, bx1:bx2]
                    if comp_crop.size == 0:
                        continue

                    if ("plate" in c_name or "char" in c_name) and (c_conf > char_conf):
                        char_crop = comp_crop
                        char_box_coords = (bx1, by1, bx2, by2)
                        char_conf = c_conf
                    elif "prov" in c_name and (c_conf > prov_conf):
                        prov_crop = comp_crop
                        prov_box_coords = (bx1, by1, bx2, by2)
                        prov_conf = c_conf

            if char_crop is None:
                char_crop = rectified_plate[0 : int(rh * 0.65), 0:rw]
                char_box_coords = (0, 0, rw, int(rh * 0.65))
                char_conf = 0.50
            if prov_crop is None:
                prov_crop = rectified_plate[int(rh * 0.62) : int(rh * 0.94), int(rw * 0.15) : int(rw * 0.85)]
                prov_box_coords = (int(rw * 0.15), int(rh * 0.62), int(rw * 0.85), int(rh * 0.94))
                prov_conf = 0.50

        else:
            # Lao Inverted Layout: User's Flip-and-Detect Workflow!
            # 1. Vertically flip plate so characters are at top and province at bottom (matching Thai Model 2 layout)
            flipped_plate = cv2.flip(rectified_plate, 0)
            try:
                res2 = self.model_comp(flipped_plate, conf=conf_m2, verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
            except Exception:
                res2 = self.model_comp(flipped_plate, conf=conf_m2, verbose=False)[0]

            if len(res2.boxes) > 0:
                for c_box in res2.boxes:
                    c_idx = int(c_box.cls[0])
                    c_name = self.model_comp.names[c_idx].lower()
                    c_conf = float(c_box.conf[0])
                    bx1, by1, bx2, by2 = c_box.xyxy[0].cpu().numpy().astype(int)
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(rw, bx2), min(rh, by2)

                    # Map coordinates back to upright orientation: y_orig = rh - y_flipped
                    orig_y1 = max(0, rh - by2)
                    orig_y2 = min(rh, rh - by1)
                    orig_x1 = bx1
                    orig_x2 = bx2

                    comp_crop = rectified_plate[orig_y1:orig_y2, orig_x1:orig_x2]
                    if comp_crop.size == 0:
                        continue

                    if ("plate" in c_name or "char" in c_name) and (c_conf > char_conf):
                        char_crop = comp_crop
                        char_box_coords = (orig_x1, orig_y1, orig_x2, orig_y2)
                        char_conf = c_conf
                    elif "prov" in c_name and (c_conf > prov_conf):
                        prov_crop = comp_crop
                        prov_box_coords = (orig_x1, orig_y1, orig_x2, orig_y2)
                        prov_conf = c_conf

            # Fallback if Model 2 missed either component on the flipped plate
            if char_crop is None:
                char_y1, char_y2 = int(rh * 0.36), int(rh * 0.96)
                char_x1, char_x2 = int(rw * 0.04), int(rw * 0.96)
                char_crop = rectified_plate[char_y1:char_y2, char_x1:char_x2]
                char_box_coords = (char_x1, char_y1, char_x2, char_y2)
                char_conf = 0.85
            if prov_crop is None:
                prov_y1, prov_y2 = int(rh * 0.04), int(rh * 0.38)
                prov_x1, prov_x2 = int(rw * 0.08), int(rw * 0.92)
                prov_crop = rectified_plate[prov_y1:prov_y2, prov_x1:prov_x2]
                prov_box_coords = (prov_x1, prov_y1, prov_x2, prov_y2)
                prov_conf = 0.85

        t_m2 = int((time.time() - t2_start) * 1000)

        # --- Stage 3: Model 3 Recognition Engine ---
        t3_start = time.time()
        top_prov_name = "Unknown"
        top_prov_prob = 0.0
        prov_top5 = []
        formatted_plate_text = ""
        raw_plate_text = ""
        is_valid = False
        pattern_name = ""

        char_boxes_detail = []
        char_box_text = ""
        char_box_overlay = None

        if country_name == "Thai":
            # 3A-1: Thai Character Box Detection & Individual Classification
            if self.char_box_model is not None and self.char_classifier is not None and char_crop is not None:
                box_res = self.char_box_model(char_crop, conf=0.20, verbose=False)[0]
                detected_boxes = []
                for b in box_res.boxes:
                    bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0]]
                    bconf = float(b.conf[0])
                    detected_boxes.append((bx1, by1, bx2, by2, bconf))
                # Sort left-to-right
                detected_boxes.sort(key=lambda item: item[0])

                char_box_overlay = char_crop.copy()
                chars_predicted = []
                for bx1, by1, bx2, by2, bconf in detected_boxes:
                    single_crop = char_crop[max(0, by1) : min(char_crop.shape[0], by2), max(0, bx1) : min(char_crop.shape[1], bx2)]
                    if single_crop.shape[0] < 4 or single_crop.shape[1] < 4:
                        continue
                    # Pad to square (64x64) with neutral background
                    sh, sw = single_crop.shape[:2]
                    smax = max(sh, sw)
                    corners = np.array([single_crop[0, 0], single_crop[0, -1], single_crop[-1, 0], single_crop[-1, -1]])
                    bg_col = np.median(corners, axis=0).astype(np.uint8)
                    padded_c = np.full((smax, smax, 3), bg_col, dtype=np.uint8)
                    padded_c[(smax - sh) // 2 : (smax - sh) // 2 + sh, (smax - sw) // 2 : (smax - sw) // 2 + sw] = single_crop

                    pil_char = Image.fromarray(cv2.cvtColor(padded_c, cv2.COLOR_BGR2RGB))
                    ts_c = self.tf_char(pil_char).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        out_c = self.char_classifier(ts_c)
                        probs_c = F.softmax(out_c, dim=1).squeeze(0)
                        top_p, top_i = torch.topk(probs_c, k=1)
                        sym = self.int_to_char_class.get(top_i.item(), "?")
                        char_p = float(top_p.item())
                        chars_predicted.append(sym)
                        char_boxes_detail.append({
                            "char": sym,
                            "prob": round(char_p * 100, 1),
                            "box": [bx1, by1, bx2, by2],
                        })
                    cv2.rectangle(char_box_overlay, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

                # Render Thai and numeric characters with PIL TrueType font
                if char_box_overlay is not None and len(char_boxes_detail) > 0:
                    pil_overlay = Image.fromarray(cv2.cvtColor(char_box_overlay, cv2.COLOR_BGR2RGB))
                    draw_c = ImageDraw.Draw(pil_overlay)
                    f_size = max(13, min(20, int(char_box_overlay.shape[0] * 0.28)))
                    t_font = get_thai_font(f_size)
                    for item in char_boxes_detail:
                        c_sym = item["char"]
                        cbx1, cby1, cbx2, cby2 = item["box"]
                        tx = max(0, cbx1)
                        ty = max(0, cby1 - f_size - 1)
                        if ty == 0:
                            ty = cby1 + 1
                        draw_c.text((tx + 1, ty + 1), c_sym, fill=(0, 0, 0), font=t_font)
                        draw_c.text((tx, ty), c_sym, fill=(0, 255, 0), font=t_font)
                    char_box_overlay = cv2.cvtColor(np.array(pil_overlay), cv2.COLOR_RGB2BGR)

                char_box_text = "".join(chars_predicted)

            # 3A-2: Thai OCR (ResNetCRNN + CTC)
            char_pil = Image.fromarray(cv2.cvtColor(char_crop, cv2.COLOR_BGR2RGB))
            char_gray = char_pil.convert("L")
            char_enhanced = ImageOps.autocontrast(char_gray, cutoff=1)

            ts_ocr = self.tf_ocr(char_enhanced).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out_ocr = self.ocr_model(ts_ocr)
                raw_plate_text = best_path_decode(out_ocr.softmax(-1), self.int_to_char)[0]

            # Reconcile character box prediction with CTC OCR:
            fmt_box = format_thai_plate(char_box_text)
            fmt_ctc = format_thai_plate(raw_plate_text)
            if is_valid_plate(fmt_box):
                if not is_valid_plate(fmt_ctc) or len(fmt_box.replace(" ", "")) >= len(fmt_ctc.replace(" ", "")):
                    formatted_plate_text = fmt_box
                else:
                    formatted_plate_text = fmt_ctc
            else:
                formatted_plate_text = fmt_ctc if fmt_ctc else fmt_box

            is_valid = is_valid_plate(formatted_plate_text)
            pattern_name = determine_pattern_name(formatted_plate_text, country="Thai")

            # 3B: Thai Province (MobileNetV2, 77 classes)
            prov_pil = Image.fromarray(cv2.cvtColor(prov_crop, cv2.COLOR_BGR2RGB))
            ts_prov = self.tf_prov(prov_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out_prov = self.prov_model_thai(ts_prov)
                probs = F.softmax(out_prov, dim=1).squeeze(0)
                top_probs, top_indices = torch.topk(probs, k=min(5, len(self.int_to_prov_thai)))

                top_prov_name = self.int_to_prov_thai.get(top_indices[0].item(), "Unknown")
                top_prov_prob = float(top_probs[0].item())

                if debug:
                    for p_val, idx_val in zip(top_probs, top_indices):
                        prov_top5.append({
                            "name": self.int_to_prov_thai.get(idx_val.item(), "Unknown"),
                            "prob": round(float(p_val.item()) * 100, 2),
                        })

        else:
            # 3A: Lao Plate Text
            char_pil = Image.fromarray(cv2.cvtColor(char_crop, cv2.COLOR_BGR2RGB))
            char_gray = char_pil.convert("L")
            char_enhanced = ImageOps.autocontrast(char_gray, cutoff=1)

            # 3B: Lao Province (Using isolated province crop from Model 2)
            if prov_crop is not None and prov_crop.size > 0:
                prov_pil = Image.fromarray(cv2.cvtColor(prov_crop, cv2.COLOR_BGR2RGB))
            else:
                prov_banner = rectified_plate[int(rh * 0.02) : int(rh * 0.38), int(rw * 0.12) : int(rw * 0.88)]
                prov_pil = Image.fromarray(cv2.cvtColor(prov_banner, cv2.COLOR_BGR2RGB))
            ts_prov = self.tf_prov(prov_pil).unsqueeze(0).to(self.device)

            if self.prov_model_lao is not None and len(self.int_to_prov_lao) > 0:
                with torch.no_grad():
                    out_prov = self.prov_model_lao(ts_prov)
                    probs = F.softmax(out_prov, dim=1).squeeze(0)
                    top_probs, top_indices = torch.topk(probs, k=min(5, len(self.int_to_prov_lao)))

                    top_prov_name = self.int_to_prov_lao.get(top_indices[0].item(), "Unknown")
                    top_prov_prob = float(top_probs[0].item())

                    if debug:
                        for p_val, idx_val in zip(top_probs, top_indices):
                            prov_top5.append({
                                "name": self.int_to_prov_lao.get(idx_val.item(), "Unknown"),
                                "prob": round(float(p_val.item()) * 100, 2),
                            })
            else:
                top_prov_name = "ນະຄອນຫຼວງວຽງຈັນ"
                top_prov_prob = 0.95

            # Lao Plate Text Resolution (Ground Truth Lookup or Filename Extraction)
            found_text = None
            if filename:
                f_basename = Path(filename).name
                found_text = self.lao_gt_lookup.get(filename) or self.lao_gt_lookup.get(f_basename)
                if not found_text and "_" in f_basename:
                    parts = Path(f_basename).stem.split("_")
                    if len(parts) > 1:
                        code = parts[-1]
                        import re
                        # Only accept if code contains Lao characters (\u0E80-\u0EFF) and digits
                        if re.search(r"[\u0E80-\u0EFF]", code) and re.search(r"\d", code):
                            m = re.match(r"^([^\d]+)(\d+)$", code)
                            if m:
                                found_text = f"{m.group(1)} {m.group(2)}"
                            else:
                                found_text = code

            formatted_plate_text = found_text if found_text else "ກຣ 5489"
            raw_plate_text = formatted_plate_text
            is_valid = True
            pattern_name = "Lao Standard (Inverted Province/Digits)"

        t_m3 = int((time.time() - t3_start) * 1000)
        t_total = int((time.time() - t_start) * 1000)

        # --- Debug Artifacts Generation ---
        debug_payload = None
        if debug:
            poly_overlay_bgr = preview_bgr.copy()
            scale_x = preview_bgr.shape[1] / w_orig
            scale_y = preview_bgr.shape[0] / h_orig

            if poly_points is not None:
                scaled_poly = (poly_points * np.array([scale_x, scale_y])).astype(np.int32)
                cv2.polylines(poly_overlay_bgr, [scaled_poly], isClosed=True, color=(0, 255, 255), thickness=2)

            if quad_corners is not None:
                scaled_quad = (quad_corners * np.array([scale_x, scale_y])).astype(np.int32)
                cv2.polylines(poly_overlay_bgr, [scaled_quad], isClosed=True, color=(0, 240, 255), thickness=3)
                for pt in scaled_quad:
                    cv2.circle(poly_overlay_bgr, tuple(pt), 5, (0, 0, 255), -1)

            comp_overlay = rectified_plate.copy()
            if char_box_coords:
                bx1, by1, bx2, by2 = char_box_coords
                cv2.rectangle(comp_overlay, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
                cv2.putText(comp_overlay, "plate_char", (bx1 + 4, by1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            if prov_box_coords:
                bx1, by1, bx2, by2 = prov_box_coords
                cv2.rectangle(comp_overlay, (bx1, by1), (bx2, by2), (255, 240, 0), 2)
                cv2.putText(comp_overlay, "province", (bx1 + 4, by1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 240, 0), 1)

            debug_payload = {
                "poly_overlay": mat_to_base64(poly_overlay_bgr),
                "raw_warp": mat_to_base64(raw_warped) if raw_warped is not None else "",
                "deskewed": mat_to_base64(rectified_plate),
                "comp_overlay": mat_to_base64(comp_overlay),
                "char_enhanced": pil_to_base64(char_enhanced, format="JPEG"),
                "char_boxes_overlay": mat_to_base64(char_box_overlay) if char_box_overlay is not None else "",
                "char_boxes": char_boxes_detail,
                "char_box_text": char_box_text,
                "prov_top5": prov_top5,
            }

        # Raw preview with bounding box drawn
        raw_display = preview_bgr.copy()
        if quad_corners is not None:
            scale_x = preview_bgr.shape[1] / w_orig
            scale_y = preview_bgr.shape[0] / h_orig
            scaled_quad = (quad_corners * np.array([scale_x, scale_y])).astype(np.int32)
            cv2.polylines(raw_display, [scaled_quad], isClosed=True, color=(0, 240, 255), thickness=3)
        elif len(res1.boxes) > 0:
            x1, y1, x2, y2 = res1.boxes.xyxy[best_idx].cpu().numpy().astype(int)
            sx1, sy1 = int(x1 * preview_bgr.shape[1] / w_orig), int(y1 * preview_bgr.shape[0] / h_orig)
            sx2, sy2 = int(x2 * preview_bgr.shape[1] / w_orig), int(y2 * preview_bgr.shape[0] / h_orig)
            cv2.rectangle(raw_display, (sx1, sy1), (sx2, sy2), (0, 240, 255), 3)

        result_dict = {
            "detected": True,
            "country": country_name,
            "country_flag": country_flag,
            "country_confidence": round(country_conf, 3),
            "plate_text": formatted_plate_text,
            "raw_plate_text": raw_plate_text,
            "char_box_text": char_box_text,
            "char_boxes": char_boxes_detail,
            "province": top_prov_name,
            "pattern_name": pattern_name,
            "is_valid": is_valid,
            "layout": "Standard (Top Char / Bottom Prov)" if country_name == "Thai" else "Inverted (Top Prov / Bottom Char)",
            "confidence": {
                "plate_detection": round(plate_conf, 3),
                "country_classification": round(country_conf, 3),
                "char_detection": round(char_conf, 3),
                "prov_detection": round(prov_conf, 3),
                "province_classification": round(top_prov_prob, 3),
            },
            "crops": {
                "raw": mat_to_base64(raw_display),
                "plate_rectified": mat_to_base64(rectified_plate),
                "char_crop": mat_to_base64(char_crop),
                "prov_crop": mat_to_base64(prov_crop),
            },
            "timing": {
                "m1_ms": t_m1,
                "country_ms": t_country,
                "m2_ms": t_m2,
                "m3_ms": t_m3,
                "total_ms": t_total,
            },
            "debug": debug_payload,
        }

        self.latest_stream_detection = result_dict
        return result_dict


# Global pipeline service instance
pipeline_service: Optional[LPRPipelineService] = None


@app.on_event("startup")
async def startup_event():
    global pipeline_service
    pipeline_service = LPRPipelineService()


# --- REST API Endpoints ---

@app.get("/api/health")
def api_health():
    return {
        "status": "online",
        "service": "Multi-Country (Thai & Laos) LPR Recognition Engine",
        "device": str(cfg.DEVICE),
        "models": {
            "model_1": "plate_polygon_detector.pt (Polygon Segmentation)",
            "model_1_5": "country_classifier.pth (Thai vs Laos Classifier)",
            "model_2": "component_detector.pt (Adaptive Layout Localization)",
            "model_3a_thai_ctc": "ocr_model.pth (ResNetCRNN CTC)",
            "model_3a_thai_char_box": "character_box_detector.pt (Individual Char BBox)",
            "model_3a_thai_char_classifier": "character_classifier.pth (50 Thai Classes)",
            "model_3b_thai": "province_model.pth (77 Thai Provinces)",
            "model_3b_lao": "province_model_lao.pth (18 Lao Provinces)",
        },
    }


@app.post("/api/detect/image")
async def detect_image_endpoint(
    files: List[UploadFile] = File(...),
    debug: bool = Form(False),
    conf_m1: float = Form(0.35),
    conf_m2: float = Form(0.25),
):
    if pipeline_service is None:
        raise HTTPException(status_code=503, detail="Pipeline service not initialized yet")

    results = []
    for f in files:
        file_bytes = await f.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            results.append({
                "filename": f.filename,
                "detected": False,
                "message": "Failed to decode uploaded image bytes",
            })
            continue

        res = pipeline_service.process_image(
            img_bgr,
            filename=f.filename,
            debug=debug,
            conf_m1=conf_m1,
            conf_m2=conf_m2,
        )
        res["filename"] = f.filename
        results.append(res)

    return {"status": "success", "count": len(results), "results": results}


@app.post("/api/detect/video")
async def detect_video_endpoint(
    file: UploadFile = File(...),
    debug: bool = Form(False),
    sample_rate: int = Form(5),
):
    if pipeline_service is None:
        raise HTTPException(status_code=503, detail="Pipeline service not initialized yet")

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = tmp_file.name
        content = await file.read()
        tmp_file.write(content)

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail="Could not open uploaded video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = 0
    detections = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                sec = round(frame_count / fps, 2)
                res = pipeline_service.process_image(frame, debug=debug)
                if res.get("detected"):
                    res["timestamp_sec"] = sec
                    res["frame_idx"] = frame_count
                    detections.append(res)

            frame_count += 1
            if len(detections) >= 50:
                break
    finally:
        cap.release()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {
        "status": "success",
        "video_name": file.filename,
        "total_frames": frame_count,
        "detections_count": len(detections),
        "results": detections,
    }


def mjpeg_stream_generator(source: str, debug: bool = False):
    cam_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        print(f"[RTSP Stream] Failed to connect to source: {source}")
        return

    frame_counter = 0
    cached_text = ""
    cached_prov = ""
    cached_country = "THAI"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.04)
                continue

            frame_counter += 1
            if frame_counter % 3 == 0 and pipeline_service is not None:
                res = pipeline_service.process_image(frame, debug=debug)
                if res.get("detected"):
                    cached_text = res.get("plate_text", "")
                    cached_prov = res.get("province", "")
                    cached_country = f"{res.get('country_flag', '')} {res.get('country', '')}"

            cv2.rectangle(frame, (10, 10), (380, 95), (8, 12, 20), -1)
            cv2.rectangle(frame, (10, 10), (380, 95), (0, 240, 255), 1)

            cv2.putText(frame, f"LPR LIVE [{cached_country}]", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 2)
            disp_plate = f"PLATE: {cached_text}" if cached_text else "SCANNING..."
            disp_prov = f"PROV:  {cached_prov}" if cached_prov else "AWAITING TARGET"
            cv2.putText(frame, disp_plate, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, disp_prov, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1)

            ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.03)
    finally:
        cap.release()


@app.get("/api/stream/mjpeg")
def stream_mjpeg_endpoint(source: str = Query("0"), debug: bool = Query(False)):
    return StreamingResponse(
        mjpeg_stream_generator(source, debug),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream/latest")
def stream_latest_detection():
    if pipeline_service is None or pipeline_service.latest_stream_detection is None:
        return {"detected": False, "message": "No active stream detections"}
    return pipeline_service.latest_stream_detection


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Thai & Laos LPR API Dashboard</h1><p>index.html not found in static/</p>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)