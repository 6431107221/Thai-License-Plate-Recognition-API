"""
src/harvest_lao_province_candidates.py

Harvests, extracts, and pseudo-labels Lao province banner crops from 3 newly added Lao datasets:
1. datasets/Lao/laos plate.v3i.yolov11
2. datasets/Lao/LicensePlateDataset
3. datasets/Lao/Lao License Plates.v2i.yolov11

Workflow:
1. Plates are cropped from YOLO annotations (box / polygon) or pre-cropped images.
2. Province banners are isolated using Model 2 flip-and-detect with robust geometric fallback.
3. Each province banner is classified using weights/province_model_lao.pth (ResNet18, 18 classes).
4. Crops are categorized into:
   datasets/Lao/lao_province_candidates/<id:02d>_<name>/
   and datasets/Lao/lao_province_candidates/_low_confidence/ (if prob < 0.35).
5. Generates candidates_summary.csv for tracking and verification.
"""

import os
import sys
import json
import glob
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ResNetProvinceClassifier
from src.preprocess import get_prov_transforms
from src.prepare_perspective_dataset import extract_quad_corners, warp_perspective_plate

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao_province_candidates"
MAP_PATH = PROJECT_ROOT / "weights" / "province_map_lao.json"
MODEL_PATH = PROJECT_ROOT / "weights" / "province_model_lao.pth"
COMP_MODEL_PATH = PROJECT_ROOT / "weights" / "component_detector.pt"
PLATE_MODEL_PATH = PROJECT_ROOT / "weights" / "plate_polygon_detector.pt"


def extract_plate_from_yolo(img: np.ndarray, lbl_path: Path) -> np.ndarray:
    """Extracts plate using YOLO bbox or polygon coordinates in label file."""
    if not lbl_path.exists():
        return None
    h, w = img.shape[:2]
    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = [l.strip().split() for l in f if l.strip()]
    if not lines:
        return None

    # Pick the largest plate annotation
    best_crop = None
    max_area = 0

    for line in lines:
        vals = [float(v) for v in line[1:]]
        if len(vals) == 4:
            xc, yc, bw, bh = vals
            x1 = max(0, int((xc - bw / 2) * w))
            y1 = max(0, int((yc - bh / 2) * h))
            x2 = min(w, int((xc + bw / 2) * w))
            y2 = min(h, int((yc + bh / 2) * h))
            area = (x2 - x1) * (y2 - y1)
            if area > max_area and (x2 > x1) and (y2 > y1):
                max_area = area
                best_crop = img[y1:y2, x1:x2]
        elif len(vals) >= 6:
            pts = np.array(vals).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            quad = extract_quad_corners(pts, img=img)
            if quad is not None:
                warped = warp_perspective_plate(img, quad, target_width=320, target_height=160)
                if warped is not None and warped.size > 0:
                    return warped
            # Fallback to bounding box of polygon
            x1, y1 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
            x2, y2 = min(w, int(pts[:, 0].max())), min(h, int(pts[:, 1].max()))
            area = (x2 - x1) * (y2 - y1)
            if area > max_area and (x2 > x1) and (y2 > y1):
                max_area = area
                best_crop = img[y1:y2, x1:x2]

    return best_crop


def extract_province_crop(plate_bgr: np.ndarray, model_comp: YOLO = None) -> np.ndarray:
    """Extracts isolated Lao province banner using flip-and-detect or geometric slice."""
    if plate_bgr is None or plate_bgr.size == 0:
        return None

    h, w = plate_bgr.shape[:2]
    # Resize to standard width if too small
    if w < 160 or h < 60:
        plate_bgr = cv2.resize(plate_bgr, (max(180, w), max(80, h)), interpolation=cv2.INTER_CUBIC)
        h, w = plate_bgr.shape[:2]

    prov_crop = None

    # Method 1: Model 2 Flip-and-Detect
    if model_comp is not None:
        try:
            flipped = cv2.flip(plate_bgr, 0)
            res = model_comp(flipped, conf=0.18, verbose=False)[0]
            for box in res.boxes:
                c_idx = int(box.cls[0])
                c_name = model_comp.names[c_idx].lower()
                if "prov" in c_name:
                    bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0]]
                    orig_y1 = max(0, h - by2)
                    orig_y2 = min(h, h - by1)
                    orig_x1 = max(0, bx1)
                    orig_x2 = min(w, bx2)
                    candidate = plate_bgr[orig_y1:orig_y2, orig_x1:orig_x2]
                    if candidate.size > 0 and candidate.shape[0] >= 10 and candidate.shape[1] >= 20:
                        prov_crop = candidate
                        break
        except Exception:
            pass

    # Method 2: Robust Geometric Slice for Lao layout (top ~38%, horizontal inset 5%-95%)
    if prov_crop is None:
        y1 = int(h * 0.03)
        y2 = int(h * 0.40)
        x1 = int(w * 0.05)
        x2 = int(w * 0.95)
        candidate = plate_bgr[y1:y2, x1:x2]
        if candidate.size > 0 and candidate.shape[0] >= 8 and candidate.shape[1] >= 16:
            prov_crop = candidate

    return prov_crop


def harvest_all_lao_provinces(conf_threshold: float = 0.35):
    print("=" * 70)
    print("   Lao Province Harvesting & Pseudo-Labeling Pipeline")
    print("=" * 70)

    # 1. Load province map
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        prov_map = json.load(f)
    int_to_prov = {int(k): v for k, v in prov_map.items()}
    num_classes = len(int_to_prov)

    # 2. Setup directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pid_str, pname in prov_map.items():
        (OUTPUT_DIR / f"{int(pid_str):02d}_{pname}").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "_low_confidence").mkdir(parents=True, exist_ok=True)

    # 3. Load models
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading Lao Province Model from: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model_lao = ResNetProvinceClassifier(n_classes=num_classes, backbone=ckpt.get("backbone", "resnet18"), pretrained=False).to(device)
    model_lao.load_state_dict(ckpt.get("model_state", ckpt))
    model_lao.eval()
    tf_prov = get_prov_transforms(is_train=False)

    print(f"Loading Component Detector from: {COMP_MODEL_PATH}")
    model_comp = YOLO(str(COMP_MODEL_PATH))

    print(f"Loading Plate Polygon Detector from: {PLATE_MODEL_PATH}")
    model_plate = YOLO(str(PLATE_MODEL_PATH))

    # 4. Gather image lists from 3 datasets
    datasets_info = [
        {
            "name": "laos_plate_v3",
            "type": "yolo_bbox",
            "base_dir": PROJECT_ROOT / "datasets" / "Lao" / "laos plate.v3i.yolov11",
            "images": list((PROJECT_ROOT / "datasets" / "Lao" / "laos plate.v3i.yolov11").glob("**/*.jpg")),
        },
        {
            "name": "license_plate_dataset",
            "type": "pre_cropped",
            "base_dir": PROJECT_ROOT / "datasets" / "Lao" / "LicensePlateDataset",
            "images": list((PROJECT_ROOT / "datasets" / "Lao" / "LicensePlateDataset").glob("**/*.jpg")),
        },
        {
            "name": "lao_license_plates_v2",
            "type": "yolo_poly",
            "base_dir": PROJECT_ROOT / "datasets" / "Lao" / "Lao License Plates.v2i.yolov11",
            "images": list((PROJECT_ROOT / "datasets" / "Lao" / "Lao License Plates.v2i.yolov11").glob("**/*.jpg")),
        },
    ]

    total_images = sum(len(d["images"]) for d in datasets_info)
    print(f"\nDiscovered {total_images} total raw images across 3 datasets:")
    for d in datasets_info:
        print(f"  - {d['name']} ({d['type']}): {len(d['images'])} images")

    records = []
    counts_by_class = {pname: 0 for pname in prov_map.values()}
    counts_by_class["_low_confidence"] = 0

    # 5. Process images
    print("\n--- Harvesting & Classifying Province Crops ---")
    pbar = tqdm(total=total_images, desc="Harvesting")

    for dinfo in datasets_info:
        ds_name = dinfo["name"]
        ds_type = dinfo["type"]
        images = dinfo["images"]

        for img_p in images:
            pbar.update(1)
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            plate = None

            # Plate extraction based on dataset type
            if ds_type in ("yolo_bbox", "yolo_poly"):
                lbl_p = Path(str(img_p).replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
                plate = extract_plate_from_yolo(img, lbl_p)
                # Fallback to model 1 if no YOLO annotation
                if plate is None or plate.size == 0:
                    try:
                        res_p = model_plate(img, conf=0.20, verbose=False)[0]
                        if len(res_p.boxes) > 0:
                            bx1, by1, bx2, by2 = res_p.boxes.xyxy[0].cpu().numpy().astype(int)
                            plate = img[by1:by2, bx1:bx2]
                    except Exception:
                        pass
            else:
                # Pre-cropped plate dataset: check if plate or full car
                h_im, w_im = img.shape[:2]
                aspect = w_im / float(max(1, h_im))
                if 1.2 <= aspect <= 4.5 and w_im <= 800:
                    plate = img
                else:
                    # Run model 1 to find plate on vehicle
                    try:
                        res_p = model_plate(img, conf=0.20, verbose=False)[0]
                        if len(res_p.boxes) > 0:
                            bx1, by1, bx2, by2 = res_p.boxes.xyxy[0].cpu().numpy().astype(int)
                            plate = img[by1:by2, bx1:bx2]
                        else:
                            plate = img
                    except Exception:
                        plate = img

            if plate is None or plate.size == 0:
                continue

            # Extract province banner crop
            prov_crop = extract_province_crop(plate, model_comp)
            if prov_crop is None or prov_crop.size == 0 or prov_crop.shape[0] < 8 or prov_crop.shape[1] < 16:
                continue

            # Classify with ResNet18 Lao model
            pil_prov = Image.fromarray(cv2.cvtColor(prov_crop, cv2.COLOR_BGR2RGB))
            ts_prov = tf_prov(pil_prov).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model_lao(ts_prov)
                probs = torch.softmax(logits, dim=-1)[0]
                top_p, top_i = torch.topk(probs, k=min(3, num_classes))
                p1_val = float(top_p[0].item())
                p1_idx = int(top_i[0].item())
                p1_name = int_to_prov.get(p1_idx, "Unknown")
                p2_val = float(top_p[1].item()) if len(top_p) > 1 else 0.0
                p2_idx = int(top_i[1].item()) if len(top_i) > 1 else -1
                p2_name = int_to_prov.get(p2_idx, "None")

            # Route to folder based on confidence
            clean_stem = img_p.stem.replace(" ", "_").replace("(", "").replace(")", "")
            out_fn = f"{ds_name}_{clean_stem}.jpg"

            if p1_val >= conf_threshold:
                target_folder = OUTPUT_DIR / f"{p1_idx:02d}_{p1_name}"
                counts_by_class[p1_name] += 1
                assigned_class = p1_name
                is_confident = True
            else:
                target_folder = OUTPUT_DIR / "_low_confidence"
                counts_by_class["_low_confidence"] += 1
                assigned_class = "_low_confidence"
                is_confident = False

            save_path = target_folder / out_fn
            cv2.imwrite(str(save_path), prov_crop)

            records.append({
                "filename": out_fn,
                "dataset_source": ds_name,
                "assigned_folder": assigned_class,
                "pred_province": p1_name,
                "pred_prov_id": p1_idx,
                "confidence": round(p1_val * 100, 2),
                "runner_up_province": p2_name,
                "runner_up_conf": round(p2_val * 100, 2),
                "margin": round((p1_val - p2_val) * 100, 2),
                "is_confident": is_confident,
                "saved_path": str(save_path.relative_to(PROJECT_ROOT)),
            })

    pbar.close()

    # 6. Save summary CSV
    df_summary = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "candidates_summary.csv"
    df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved candidates metadata to: {csv_path}")

    # 7. Print summary report
    print("\n" + "=" * 60)
    print("   Lao Province Candidates Extraction Summary")
    print("=" * 60)
    print(f"{'Class ID':<10} {'Province Name':<28} {'Harvested Crops':<15}")
    print("-" * 60)
    total_confident = 0
    for pid_str, pname in prov_map.items():
        cnt = counts_by_class.get(pname, 0)
        total_confident += cnt
        print(f"{int(pid_str):<10} {pname:<28} {cnt:<15}")
    print("-" * 60)
    print(f"{'--':<10} {'_low_confidence':<28} {counts_by_class['_low_confidence']:<15}")
    print(f"\nTotal Crops Extracted: {len(records)}")
    print(f"Total Confident (>= {conf_threshold*100:.0f}%): {total_confident}")
    print(f"Total Low Confidence: {counts_by_class['_low_confidence']}")
    print("=" * 60)


if __name__ == "__main__":
    harvest_all_lao_provinces()
