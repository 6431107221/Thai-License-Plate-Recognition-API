"""
src/api_server.py

Comprehensive Thai License Plate Recognition API Service and Web Dashboard Server.
Integrates 4 models:
  - Model 1: Plate Polygon Segmentation + Perspective Transformation + Fine Deskew
  - Model 2: Plate Character & Province Component Bounding Box Detection
  - Model 3A: ResNetCRNN + CTC for Thai Alphanumeric Plate Text Recognition
  - Model 3B: MobileNetV2 for 77 Thai Provinces Classification

Features:
  - File Upload Mode: Single image, batch/multiple images, and video files
  - Live RTSP Stream / Webcam Mode with MJPEG real-time feed
  - 3-Stage Visual Pipeline Breakdown: Raw -> Model 1 -> Model 2 -> Model 3
  - Debug Mode ON/OFF: On generates diagnostic overlays/profiles; Off skips generation to save resources
  - Strict Thai plate format validation (NCC NNNN, CC NNNN, C NNNN, NN-NNNN, NNNNN)
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
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from fastapi import FastAPI, UploadFile, File, Form, Query, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.config import cfg
from src.models import ResNetCRNN, ProvinceClassifier, best_path_decode
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
    title="Thai License Plate Recognition (Thai LPR)",
    description="Multi-stage Deep Learning Pipeline for Thai Car Plate Recognition",
    version="2.0.0",
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


def determine_pattern_name(text: str) -> str:
    """Determines the specific Thai plate pattern type."""
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
        m2_path = cfg.WEIGHTS_DIR / "component_detector.pt"
        ocr_path = cfg.WEIGHTS_DIR / "ocr_model.pth"
        prov_path = cfg.WEIGHTS_DIR / "province_model.pth"
        char_map_path = cfg.WEIGHTS_DIR / "int_to_char.json"
        prov_map_path = cfg.WEIGHTS_DIR / "province_map.json"

        # 1. Load Model 1 (Plate Polygon Segmentation)
        print(f"[Model 1] Loading plate polygon detector from: {m1_path}")
        self.model_plate = YOLO(str(m1_path))

        # 2. Load Model 2 (Component Detector: plate_char & province)
        print(f"[Model 2] Loading component detector from: {m2_path}")
        self.model_comp = YOLO(str(m2_path))

        # 3. Load Model 3A (OCR Model)
        print(f"[Model 3A] Loading ResNetCRNN OCR model from: {ocr_path}")
        self.ocr_model, self.int_to_char = self._load_ocr_model(ocr_path, char_map_path)

        # 4. Load Model 3B (Province Model)
        print(f"[Model 3B] Loading MobileNetV2 Province model from: {prov_path}")
        self.prov_model, self.int_to_prov = self._load_prov_model(prov_path, prov_map_path)

        # 5. Transforms
        self.tf_ocr = get_ocr_transforms(is_train=False)
        self.tf_prov = get_prov_transforms(is_train=False)

        # Global latest detection for RTSP stream viewer
        self.latest_stream_detection: Optional[Dict[str, Any]] = None

        print("[LPRPipelineService] All 4 Models Successfully Loaded & Ready!")

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

        model = ProvinceClassifier(n_classes=len(int_to_prov), pretrained=False).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state_dict)
        model.eval()
        return model, int_to_prov

    def process_image(
        self,
        img_bgr: np.ndarray,
        debug: bool = False,
        conf_m1: float = 0.35,
        conf_m2: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end 3-stage recognition pipeline on an OpenCV BGR image:
        Raw -> Model 1 (Plate Polygon & Rectification)
            -> Model 2 (Plate Characters & Province Bounding Boxes)
            -> Model 3 (OCR & Province Engine)
        """
        t_start = time.time()
        h_orig, w_orig = img_bgr.shape[:2]

        # Stage 0: Raw Preview
        # Resize raw preview if image is massive (for snappy UI rendering)
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

        t_m1 = int((time.time() - t1_start) * 1000)

        if len(res1.boxes) == 0:
            return {
                "detected": False,
                "message": "No vehicle license plate detected in image",
                "timing": {"m1_ms": t_m1, "m2_ms": 0, "m3_ms": 0, "total_ms": int((time.time() - t_start) * 1000)},
                "raw_preview": mat_to_base64(preview_bgr),
                "debug": None,
            }

        # Select highest-confidence plate
        confidences = res1.boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confidences))
        plate_conf = float(confidences[best_idx])

        rectified_plate = None
        raw_warped = None
        quad_corners = None
        poly_points = None

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

        # Fallback to standard bounding box crop if polygon or quad failed
        if rectified_plate is None:
            x1, y1, x2, y2 = res1.boxes.xyxy[best_idx].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)
            raw_box = img_bgr[y1:y2, x1:x2]
            if raw_box.size > 0:
                rectified_plate = cv2.resize(raw_box, (320, 160), interpolation=cv2.INTER_CUBIC)
                raw_warped = rectified_plate.copy()

        if rectified_plate is None or rectified_plate.size == 0:
            return {
                "detected": False,
                "message": "Failed to extract or rectify detected license plate",
                "timing": {"m1_ms": t_m1, "m2_ms": 0, "m3_ms": 0, "total_ms": int((time.time() - t_start) * 1000)},
                "raw_preview": mat_to_base64(preview_bgr),
                "debug": None,
            }

        rh, rw = rectified_plate.shape[:2]

        # --- Stage 2: Model 2 Component Detection (plate_char & province) ---
        t2_start = time.time()
        try:
            res2 = self.model_comp(rectified_plate, conf=conf_m2, verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
        except Exception:
            res2 = self.model_comp(rectified_plate, conf=conf_m2, verbose=False)[0]

        t_m2 = int((time.time() - t2_start) * 1000)

        char_crop = None
        prov_crop = None
        char_box_coords = None
        prov_box_coords = None
        char_conf = 0.0
        prov_conf = 0.0

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

        # Standard geometric fallbacks if Model 2 didn't detect either component
        if char_crop is None:
            char_crop = rectified_plate[0 : int(rh * 0.65), 0:rw]
            char_box_coords = (0, 0, rw, int(rh * 0.65))
            char_conf = 0.50
        if prov_crop is None:
            prov_crop = rectified_plate[int(rh * 0.60) : rh, 0:rw]
            prov_box_coords = (0, int(rh * 0.60), rw, rh)
            prov_conf = 0.50

        # --- Stage 3: Model 3 Recognition Engine ---
        t3_start = time.time()

        # 3A: Model 3A (OCR ResNetCRNN + CTC)
        char_pil = Image.fromarray(cv2.cvtColor(char_crop, cv2.COLOR_BGR2RGB))
        char_gray = char_pil.convert("L")
        char_enhanced = ImageOps.autocontrast(char_gray, cutoff=1)

        ts_ocr = self.tf_ocr(char_enhanced).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out_ocr = self.ocr_model(ts_ocr)
            raw_plate_text = best_path_decode(out_ocr.softmax(-1), self.int_to_char)[0]

        formatted_plate_text = format_thai_plate(raw_plate_text)
        is_valid = is_valid_plate(formatted_plate_text)
        pattern_name = determine_pattern_name(formatted_plate_text)

        # 3B: Model 3B (Province Classifier MobileNetV2)
        prov_pil = Image.fromarray(cv2.cvtColor(prov_crop, cv2.COLOR_BGR2RGB))
        ts_prov = self.tf_prov(prov_pil).unsqueeze(0).to(self.device)

        top_prov_name = "Unknown"
        top_prov_prob = 0.0
        prov_top5 = []

        with torch.no_grad():
            out_prov = self.prov_model(ts_prov)
            probs = F.softmax(out_prov, dim=1).squeeze(0)
            top_probs, top_indices = torch.topk(probs, k=min(5, len(self.int_to_prov)))

            top_prov_name = self.int_to_prov.get(top_indices[0].item(), "Unknown")
            top_prov_prob = float(top_probs[0].item())

            if debug:
                for p_val, idx_val in zip(top_probs, top_indices):
                    prov_top5.append({
                        "name": self.int_to_prov.get(idx_val.item(), "Unknown"),
                        "prob": round(float(p_val.item()) * 100, 2),
                    })

        t_m3 = int((time.time() - t3_start) * 1000)
        t_total = int((time.time() - t_start) * 1000)

        # --- Debug Artifacts Generation (Strictly conditional on debug=True) ---
        debug_payload = None
        if debug:
            # 1. Raw image with polygon and quad contour
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

            # 2. Rectified Plate with Model 2 component boxes
            comp_overlay = rectified_plate.copy()
            if char_box_coords:
                bx1, by1, bx2, by2 = char_box_coords
                cv2.rectangle(comp_overlay, (bx1, by1), (bx2, by2), (0, 165, 255), 2)  # Amber
                cv2.putText(comp_overlay, "plate_char", (bx1 + 4, by1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            if prov_box_coords:
                bx1, by1, bx2, by2 = prov_box_coords
                cv2.rectangle(comp_overlay, (bx1, by1), (bx2, by2), (255, 240, 0), 2)  # Cyan
                cv2.putText(comp_overlay, "province", (bx1 + 4, by1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 240, 0), 1)

            debug_payload = {
                "poly_overlay": mat_to_base64(poly_overlay_bgr),
                "raw_warp": mat_to_base64(raw_warped) if raw_warped is not None else "",
                "deskewed": mat_to_base64(rectified_plate),
                "comp_overlay": mat_to_base64(comp_overlay),
                "char_enhanced": pil_to_base64(char_enhanced, format="JPEG"),
                "prov_top5": prov_top5,
            }

        # Raw preview with bounding box drawn for Stage 0
        raw_display = preview_bgr.copy()
        if quad_corners is not None:
            scale_x = preview_bgr.shape[1] / w_orig
            scale_y = preview_bgr.shape[0] / h_orig
            scaled_quad = (quad_corners * np.array([scale_x, scale_y])).astype(np.int32)
            cv2.polylines(raw_display, [scaled_quad], isClosed=True, color=(0, 240, 255), thickness=3)
        else:
            x1, y1, x2, y2 = res1.boxes.xyxy[best_idx].cpu().numpy().astype(int)
            sx1, sy1 = int(x1 * preview_bgr.shape[1] / w_orig), int(y1 * preview_bgr.shape[0] / h_orig)
            sx2, sy2 = int(x2 * preview_bgr.shape[1] / w_orig), int(y2 * preview_bgr.shape[0] / h_orig)
            cv2.rectangle(raw_display, (sx1, sy1), (sx2, sy2), (0, 240, 255), 3)

        result_dict = {
            "detected": True,
            "plate_text": formatted_plate_text,
            "raw_plate_text": raw_plate_text,
            "province": top_prov_name,
            "pattern_name": pattern_name,
            "is_valid": is_valid,
            "confidence": {
                "plate_detection": round(plate_conf, 3),
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
                "m2_ms": t_m2,
                "m3_ms": t_m3,
                "total_ms": t_total,
            },
            "debug": debug_payload,
        }

        # Cache for RTSP live stream HUD
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
        "service": "Thai LPR Recognition Engine",
        "device": str(cfg.DEVICE),
        "models": {
            "model_1": "plate_polygon_detector.pt (Polygon Segmentation)",
            "model_2": "component_detector.pt (Character & Province Detection)",
            "model_3a": "ocr_model.pth (ResNetCRNN CTC)",
            "model_3b": "province_model.pth (MobileNetV2 77 Provinces)",
        },
    }


@app.post("/api/detect/image")
async def detect_image_endpoint(
    files: List[UploadFile] = File(...),
    debug: bool = Form(False),
    conf_m1: float = Form(0.35),
    conf_m2: float = Form(0.25),
):
    """
    Accepts single or batch image uploads. Runs the 3-stage recognition pipeline.
    Respects debug toggle to generate or skip resource-heavy breakdown images.
    """
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
    sample_rate: int = Form(5),  # Sample every Nth frame
):
    """
    Processes an uploaded video (.mp4, .mov, .avi) by sampling frames,
    running 3-stage recognition, and compiling a chronological detection timeline.
    """
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
            if len(detections) >= 50:  # Cap at 50 detections to avoid payload explosion
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


# --- RTSP / Live Video MJPEG Stream Generator ---

def mjpeg_stream_generator(source: str, debug: bool = False):
    """
    Captures live frames from RTSP URL or local camera index,
    runs inference, annotates HUD onto frame, and yields MJPEG stream.
    """
    # Check if source is numeric camera index
    cam_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        print(f"[RTSP Stream] Failed to connect to source: {source}")
        return

    frame_counter = 0
    cached_box = None
    cached_text = ""
    cached_prov = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.04)
                continue

            frame_counter += 1
            # Run detection every 3rd frame for real-time smoothness
            if frame_counter % 3 == 0 and pipeline_service is not None:
                res = pipeline_service.process_image(frame, debug=debug)
                if res.get("detected"):
                    cached_text = res.get("plate_text", "")
                    cached_prov = res.get("province", "")

            # Draw HUD overlay
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (10, 10), (360, 85), (8, 12, 20), -1)
            cv2.rectangle(frame, (10, 10), (360, 85), (0, 240, 255), 1)

            cv2.putText(frame, "THAI LPR LIVE STREAM", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 2)
            disp_plate = f"PLATE: {cached_text}" if cached_text else "SCANNING..."
            disp_prov = f"PROV:  {cached_prov}" if cached_prov else "AWAITING TARGET"
            cv2.putText(frame, disp_plate, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, disp_prov, (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.03)  # ~30 FPS
    finally:
        cap.release()


@app.get("/api/stream/mjpeg")
def stream_mjpeg_endpoint(source: str = Query("0"), debug: bool = Query(False)):
    """
    Streams live annotated MJPEG video from an RTSP port or local webcam.
    Example: `/api/stream/mjpeg?source=rtsp://192.168.1.50:554/live&debug=false`
    """
    return StreamingResponse(
        mjpeg_stream_generator(source, debug),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream/latest")
def stream_latest_detection():
    """Returns the most recent detection result from the live stream."""
    if pipeline_service is None or pipeline_service.latest_stream_detection is None:
        return {"detected": False, "message": "No active stream detections"}
    return pipeline_service.latest_stream_detection


# --- Static Files & Web Dashboard ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Thai LPR API Dashboard</h1><p>index.html not found in static/</p>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)