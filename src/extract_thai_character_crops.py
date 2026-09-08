"""
src/extract_thai_character_crops.py

Extracts individual Thai character bounding boxes from YOLO annotations in
'datasets/LPR 2 - Character Box Detection.yolov11',
maps them left-to-right to the Ground Truth text in 'output/ground_truth_crops/gt_plate_char.csv',
and organizes them into character folders (e.g. 1, 2, 3, 4, ก, ข, ค...).
"""

import os
import re
import shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROBOFLOW_DIR = PROJECT_ROOT / "datasets" / "LPR 2 - Character Box Detection.yolov11"
GT_CSV_PATH = PROJECT_ROOT / "output" / "ground_truth_crops" / "gt_plate_char.csv"
OUTPUT_BASE = PROJECT_ROOT / "datasets" / "thai_character_crops"

DIR_RAW = OUTPUT_BASE / "by_character"
DIR_SQUARE = OUTPUT_BASE / "by_character_square"
DIR_SPLITS = OUTPUT_BASE / "splits"
METADATA_CSV = OUTPUT_BASE / "metadata.csv"


def make_square_padded(crop_bgr: np.ndarray, target_size: int = 64) -> np.ndarray:
    """
    Pads a cropped character to a square canvas with aspect ratio preserved,
    using an adaptive neutral background estimated from the crop edges.
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_size, target_size, 3), 255, dtype=np.uint8)

    max_dim = max(h, w)
    # Estimate background color from the 4 corners of the crop
    corners = np.array([crop_bgr[0, 0], crop_bgr[0, -1], crop_bgr[-1, 0], crop_bgr[-1, -1]])
    bg_color = np.median(corners, axis=0).astype(np.uint8)

    padded = np.full((max_dim, max_dim, 3), bg_color, dtype=np.uint8)
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    padded[y_offset : y_offset + h, x_offset : x_offset + w] = crop_bgr

    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def run_extraction():
    print("================================================================")
    print("THAI CHARACTER DATASET EXTRACTION & GROUND TRUTH MAPPING")
    print("================================================================")

    # 1. Load Ground Truth CSV
    print(f"\n[1/5] Loading Ground Truth: {GT_CSV_PATH}")
    df_gt = pd.read_csv(GT_CSV_PATH, skiprows=1)
    tag_lookup = {}
    for _, row in df_gt.iterrows():
        if pd.isna(row["tag_id"]):
            continue
        tid = int(row["tag_id"])
        raw_gt = str(row["gt_plate"]).strip() if pd.notna(row["gt_plate"]) else ""
        clean_chars = [c for c in raw_gt if c not in [" ", "-", "–"]]
        tag_lookup[tid] = {
            "raw_gt": raw_gt,
            "clean_chars": clean_chars,
            "image": str(row["image"]) if pd.notna(row["image"]) else "",
        }
    print(f"      Loaded {len(tag_lookup)} plate records from CSV.")

    # 2. Collect Roboflow image-label pairs
    print(f"\n[2/5] Scanning annotations in: {ROBOFLOW_DIR}")
    rf_items = []
    for split_name in ["train", "valid", "test"]:
        split_dir = ROBOFLOW_DIR / split_name
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.*")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            rf_items.append({
                "split": split_name,
                "image_path": img_path,
                "label_path": lbl_path,
            })
    print(f"      Found {len(rf_items)} annotated images across train/valid/test.")

    # 3. Setup output directories
    for d in [DIR_RAW, DIR_SQUARE, DIR_SPLITS]:
        d.mkdir(parents=True, exist_ok=True)

    records = []
    skipped_no_gt = 0
    skipped_zero_boxes = 0
    skipped_mismatch = 0
    matched_plates = 0
    char_counter = Counter()

    print("\n[3/5] Slicing characters and mapping to ground truth...")
    char_crop_id = 0

    for item in tqdm(rf_items, desc="Extracting"):
        img_path = item["image_path"]
        lbl_path = item["label_path"]
        split_name = item["split"]

        m = re.match(r"^(\d{6})_", img_path.name)
        tid = int(m.group(1)) if m else None

        gt_info = tag_lookup.get(tid)
        if not gt_info or not gt_info["clean_chars"]:
            skipped_no_gt += 1
            continue

        clean_chars = gt_info["clean_chars"]
        raw_gt = gt_info["raw_gt"]

        if not lbl_path.exists():
            skipped_zero_boxes += 1
            continue

        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = [float(x) for x in line.strip().split()]
                if len(parts) >= 5:
                    boxes.append(parts)

        if len(boxes) == 0:
            skipped_zero_boxes += 1
            continue

        # Sort boxes strictly from left to right by x_center
        boxes.sort(key=lambda b: b[1])

        # Handle known edge cases:
        if tid == 84 and len(boxes) == 7 and len(clean_chars) == 6:
            # Filter out Box 2 (dash with height 0.176)
            del boxes[2]
        elif tid == 455 and len(boxes) == 7 and len(clean_chars) == 6:
            # Filter out Box 0 (left-edge border artifact at x=0)
            del boxes[0]

        if len(boxes) != len(clean_chars):
            skipped_mismatch += 1
            continue

        matched_plates += 1

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h_img, w_img = img_bgr.shape[:2]

        for box_idx, (box, ch) in enumerate(zip(boxes, clean_chars)):
            char_crop_id += 1
            cls_id, xc, yc, bw, bh = box[:5]
            x1 = max(0, int((xc - bw / 2.0) * w_img))
            y1 = max(0, int((yc - bh / 2.0) * h_img))
            x2 = min(w_img, int((xc + bw / 2.0) * w_img))
            y2 = min(h_img, int((yc + bh / 2.0) * h_img))

            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            square_crop = make_square_padded(crop, target_size=64)

            # Folder naming (use safe character folder)
            char_folder = ch

            # 1. By character raw
            ch_raw_dir = DIR_RAW / char_folder
            ch_raw_dir.mkdir(parents=True, exist_ok=True)
            crop_filename = f"{tid:06d}_box{box_idx}_{ch}_{char_crop_id}.jpg"
            raw_path = ch_raw_dir / crop_filename
            cv2.imwrite(str(raw_path), crop)

            # 2. By character square padded
            ch_sq_dir = DIR_SQUARE / char_folder
            ch_sq_dir.mkdir(parents=True, exist_ok=True)
            sq_path = ch_sq_dir / crop_filename
            cv2.imwrite(str(sq_path), square_crop)

            # 3. By split for PyTorch ImageFolder
            split_ch_dir = DIR_SPLITS / split_name / char_folder
            split_ch_dir.mkdir(parents=True, exist_ok=True)
            split_path = split_ch_dir / crop_filename
            cv2.imwrite(str(split_path), square_crop)

            char_counter[ch] += 1

            records.append({
                "char_crop_id": char_crop_id,
                "character": ch,
                "split": split_name,
                "tag_id": tid,
                "raw_gt_plate": raw_gt,
                "box_index": box_idx,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "crop_width": x2 - x1,
                "crop_height": y2 - y1,
                "source_image": img_path.name,
                "rel_path_raw": f"by_character/{char_folder}/{crop_filename}",
                "rel_path_square": f"by_character_square/{char_folder}/{crop_filename}",
                "rel_path_split": f"splits/{split_name}/{char_folder}/{crop_filename}",
            })

    # 4. Save metadata CSV
    print(f"\n[4/5] Saving metadata CSV: {METADATA_CSV}")
    df_meta = pd.DataFrame(records)
    df_meta.to_csv(METADATA_CSV, index=False, encoding="utf-8")
    print(f"      Saved {len(df_meta)} character crop records.")

    # 5. Summary Report
    print("\n[5/5] EXTRACTION SUMMARY:")
    print("----------------------------------------------------------------")
    print(f"  - Successfully matched plates: {matched_plates}")
    print(f"  - Total character crops created: {len(records)}")
    print(f"  - Distinct character classes: {len(char_counter)}")
    print(f"  - Plates skipped (unannotated / no GT text): {skipped_no_gt + skipped_zero_boxes}")
    if skipped_mismatch:
        print(f"  - Plates skipped (box count mismatch): {skipped_mismatch}")
    print("----------------------------------------------------------------")
    print("\nCharacter Class Distribution:")
    for ch, cnt in sorted(char_counter.items()):
        print(f"  '{ch}': {cnt:4d} samples")
    print("================================================================")
    print(f"Output folders created in: {OUTPUT_BASE}")
    print(f"  1. by_character/        (Raw crops organized by letter/number)")
    print(f"  2. by_character_square/ (Square padded 64x64 for CNN classifier)")
    print(f"  3. splits/              (train/valid/test ImageFolder structure)")
    print(f"  4. metadata.csv         (Full traceability metadata)")
    print("================================================================")


if __name__ == "__main__":
    run_extraction()
