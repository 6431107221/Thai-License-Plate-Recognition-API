"""
src/harvest_candidate_character_crops.py

Automated character harvesting & pseudo-labeling pipeline (Approach A).
Processes the 3,219 license plate images from
'datasets/thai-car-license-plate-province.v5i.yolov11':
  1. Detects 'plate_char' using Model 2 (weights/component_detector.pt).
  2. Runs Model 3A (weights/ocr_model.pth) to read the full character sequence.
  3. Detects individual character bounding boxes using weights/character_box_detector.pt.
  4. Sorts boxes left-to-right and pairs them with the OCR string.
  5. Slices each character, pads to square (64x64) with neutral background.
  6. Saves candidate crops into 'datasets/thai_character_crops/candidates/<char>/'.
  7. Generates a metadata table and verification report for user review.
"""

import os
import sys
import re
import json
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

from src.models import ResNetCRNN, best_path_decode
from src.preprocess import get_ocr_transforms

# Local fallback for make_square_padded if not in preprocess
def pad_to_square(crop_bgr: np.ndarray, target_size: int = 64) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    max_dim = max(h, w)
    corners = np.array([crop_bgr[0, 0], crop_bgr[0, -1], crop_bgr[-1, 0], crop_bgr[-1, -1]])
    bg_color = np.median(corners, axis=0).astype(np.uint8)
    padded = np.full((max_dim, max_dim, 3), bg_color, dtype=np.uint8)
    y_off = (max_dim - h) // 2
    x_off = (max_dim - w) // 2
    padded[y_off : y_off + h, x_off : x_off + w] = crop_bgr
    return cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)


def clean_ocr_text(raw_text: str):
    """Removes dashes, spaces, and unrecognized punctuation."""
    cleaned = []
    for ch in raw_text:
        if ch in " -–—_·.":
            continue
        cleaned.append(ch)
    return cleaned


def harvest_candidates(max_plates=None):
    print("==================================================================")
    print("APPROACH A: AUTO-CROP & PSEUDO-LABELING CHARACTER HARVESTER")
    print("==================================================================")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    weights_dir = PROJECT_ROOT / "weights"
    comp_pt = weights_dir / "component_detector.pt"
    char_box_pt = weights_dir / "character_box_detector.pt"
    ocr_pt = weights_dir / "ocr_model.pth"
    char_map_json = weights_dir / "int_to_char.json"

    if not comp_pt.exists():
        print(f"[Error] {comp_pt} not found.")
        return
    if not char_box_pt.exists():
        print(f"[Error] {char_box_pt} not found.")
        return
    if not ocr_pt.exists() or not char_map_json.exists():
        print(f"[Error] OCR weights or char map not found.")
        return

    # 1. Load Models
    print("\n[1/4] Loading Detection & OCR Models...")
    comp_model = YOLO(str(comp_pt))
    box_model = YOLO(str(char_box_pt))

    with open(char_map_json, "r", encoding="utf-8") as f:
        int_to_char = {int(k): v for k, v in json.load(f).items()}

    ocr_model = ResNetCRNN(img_channel=1, num_classes=len(int_to_char)).to(device)
    ckpt = torch.load(ocr_pt, map_location=device)
    ocr_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    ocr_model.eval()

    tf_ocr = get_ocr_transforms(is_train=False)

    # 2. Gather Images
    plates_dir = PROJECT_ROOT / "datasets" / "thai-car-license-plate-province.v5i.yolov11"
    all_imgs = []
    for split in ["train", "valid", "test"]:
        split_imgs = list((plates_dir / split / "images").glob("*.jpg")) + list((plates_dir / split / "images").glob("*.png"))
        all_imgs.extend(split_imgs)

    if max_plates:
        all_imgs = all_imgs[:max_plates]

    print(f"[2/4] Total source license plate images: {len(all_imgs)}")

    output_dir = PROJECT_ROOT / "datasets" / "thai_character_crops" / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Processing Loop
    print("\n[3/4] Harvesting & Slicing Characters...")
    harvested_records = []
    matched_plates = 0
    total_crops = 0

    for img_p in tqdm(all_imgs, desc="Harvesting"):
        plate_img = cv2.imread(str(img_p))
        if plate_img is None:
            continue
        ph, pw = plate_img.shape[:2]

        # Step A: Detect 'plate_char'
        comp_res = comp_model(plate_img, conf=0.25, verbose=False)[0]
        char_region_box = None
        for b in comp_res.boxes:
            if comp_model.names[int(b.cls[0])] == "plate_char":
                char_region_box = [int(v) for v in b.xyxy[0]]
                break

        if char_region_box is None:
            # Fallback: top 70% of plate
            char_crop = plate_img[0 : int(ph * 0.70), 0:pw]
            x_offset, y_offset = 0, 0
        else:
            x1, y1, x2, y2 = char_region_box
            char_crop = plate_img[max(0, y1) : min(ph, y2), max(0, x1) : min(pw, x2)]
            x_offset, y_offset = x1, y1

        if char_crop.shape[0] < 10 or char_crop.shape[1] < 10:
            continue

        # Step B: OCR string prediction on plate_char
        pil_crop = Image.fromarray(cv2.cvtColor(char_crop, cv2.COLOR_BGR2RGB))
        ts = tf_ocr(pil_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = ocr_model(ts)
            log_probs = torch.nn.functional.log_softmax(preds, dim=-1)
            raw_text = best_path_decode(log_probs, int_to_char)[0]

        cleaned_chars = clean_ocr_text(raw_text)
        if len(cleaned_chars) < 2:
            continue

        # Step C: Detect individual character boxes inside plate_char
        box_res = box_model(char_crop, conf=0.20, verbose=False)[0]
        detected_boxes = []
        for b in box_res.boxes:
            bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0]]
            conf = float(b.conf[0])
            detected_boxes.append((bx1, by1, bx2, by2, conf))

        # Sort left-to-right
        detected_boxes.sort(key=lambda item: item[0])

        # Step D: Match boxes with OCR characters
        # Exact match count gives the highest confidence
        if len(detected_boxes) == len(cleaned_chars):
            matched_plates += 1
            for idx, ((bx1, by1, bx2, by2, conf), char_lbl) in enumerate(zip(detected_boxes, cleaned_chars)):
                # Crop character
                ch_crop = char_crop[max(0, by1) : min(char_crop.shape[0], by2), max(0, bx1) : min(char_crop.shape[1], bx2)]
                if ch_crop.shape[0] < 5 or ch_crop.shape[1] < 5:
                    continue

                square_crop = pad_to_square(ch_crop, target_size=64)

                char_folder = output_dir / char_lbl
                char_folder.mkdir(parents=True, exist_ok=True)

                filename = f"auto_{img_p.stem}_c{idx}_{char_lbl}.jpg"
                crop_path = char_folder / filename
                cv2.imwrite(str(crop_path), square_crop)

                total_crops += 1
                harvested_records.append({
                    "filename": filename,
                    "character": char_lbl,
                    "confidence": round(conf, 3),
                    "source_plate": img_p.name,
                    "full_ocr_text": raw_text,
                    "crop_path": str(crop_path.relative_to(PROJECT_ROOT)),
                })

    # 4. Save Manifest
    print(f"\n[4/4] Saving Candidate Metadata...")
    meta_df = pd.DataFrame(harvested_records)
    meta_path = output_dir.parent / "candidates_metadata.csv"
    meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")

    print("\n==================================================================")
    print("HARVESTING COMPLETE!")
    print(f"Total Plates Processed: {len(all_imgs)}")
    print(f"Plates with 100% Exact Box-to-Character Match: {matched_plates}")
    print(f"Total Candidate Character Crops Harvested: {total_crops}")
    print(f"Unique Characters Harvested: {meta_df['character'].nunique() if not meta_df.empty else 0}")
    print(f"Candidates Saved in: {output_dir}")
    print(f"Metadata Manifest: {meta_path}")
    print("==================================================================\n")

    if not meta_df.empty:
        print("Top 10 Harvested Characters:")
        print(meta_df["character"].value_counts().head(10))
        rare = meta_df["character"].value_counts().tail(10)
        print("\nLeast Harvested Characters:")
        print(rare)


if __name__ == "__main__":
    harvest_candidates()
