"""
src/extract_thai_province_crops.py

Extracts Thai license plate province crops from:
  1. datasets/thai-car-license-plate-province.v5i.yolov11 (3,219 plates, 77 provinces)
  2. output/ground_truth_crops/gt_province.csv (798 crops)

Organizes crops into standard ImageFolder splits:
  datasets/thai_province_crops/
    train/<province_id>_<thai_name>/
    valid/<province_id>_<thai_name>/
    test/<province_id>_<thai_name>/

Generates:
  datasets/thai_province_crops/metadata.csv
"""

import os
import json
import yaml
import shutil
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_PROV_DIR = PROJECT_ROOT / "datasets" / "thai-car-license-plate-province.v5i.yolov11"
GT_PROV_CSV = PROJECT_ROOT / "output" / "ground_truth_crops" / "gt_province.csv"
GT_PROV_DIR = PROJECT_ROOT / "output" / "ground_truth_crops"

ABBR_MAP_PATH = PROJECT_ROOT / "weights" / "province_abbr_map.json"
PROV_MAP_PATH = PROJECT_ROOT / "weights" / "province_map.json"

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "thai_province_crops"


def load_mappings():
    with open(ABBR_MAP_PATH, "r", encoding="utf-8") as f:
        abbr_map = json.load(f)
    with open(PROV_MAP_PATH, "r", encoding="utf-8") as f:
        prov_map = json.load(f)
    name_to_id = {v: int(k) for k, v in prov_map.items()}
    return abbr_map, prov_map, name_to_id


def extract_from_roboflow(abbr_map, metadata_rows):
    data_yaml_path = YOLO_PROV_DIR / "data.yaml"
    if not data_yaml_path.exists():
        print(f"[Warning] {data_yaml_path} not found.")
        return

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        ydata = yaml.safe_load(f)
    class_names = ydata["names"]

    for split in ["train", "valid", "test"]:
        lbl_dir = YOLO_PROV_DIR / split / "labels"
        img_dir = YOLO_PROV_DIR / split / "images"
        lbl_files = sorted(list(lbl_dir.glob("*.txt")))

        print(f"Extracting {split} split ({len(lbl_files)} label files)...")
        for lf in tqdm(lbl_files, desc=f"Roboflow {split}"):
            stem = lf.stem
            img_p = img_dir / f"{stem}.jpg"
            if not img_p.exists():
                img_p = img_dir / f"{stem}.png"
            if not img_p.exists():
                continue

            img = cv2.imread(str(img_p))
            if img is None:
                continue
            h, w = img.shape[:2]

            with open(lf, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]

            for box_idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 5:
                    continue

                cid = int(parts[0])
                if cid >= len(class_names):
                    continue

                abbr = class_names[cid]
                if abbr not in abbr_map:
                    continue

                prov_info = abbr_map[abbr]
                prov_id = prov_info["province_id"]
                thai_name = prov_info["thai_name"]

                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = max(0, int((cx - bw / 2) * w))
                y1 = max(0, int((cy - bh / 2) * h))
                x2 = min(w, int((cx + bw / 2) * w))
                y2 = min(h, int((cy + bh / 2) * h))

                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue

                crop = img[y1:y2, x1:x2]
                folder_name = f"{prov_id:02d}_{thai_name}"
                target_folder = OUTPUT_DIR / split / folder_name
                target_folder.mkdir(parents=True, exist_ok=True)

                crop_filename = f"rf_{split}_{stem}_b{box_idx}.jpg"
                crop_path = target_folder / crop_filename
                cv2.imwrite(str(crop_path), crop)

                metadata_rows.append({
                    "filename": crop_filename,
                    "rel_path": str(crop_path.relative_to(OUTPUT_DIR)),
                    "split": split,
                    "source": "roboflow_v5i",
                    "province_id": prov_id,
                    "thai_name": thai_name,
                    "abbr": abbr,
                    "width": crop.shape[1],
                    "height": crop.shape[0],
                })


def extract_from_gt_province(name_to_id, metadata_rows):
    if not GT_PROV_CSV.exists():
        print(f"[Warning] {GT_PROV_CSV} not found.")
        return

    # Check header
    with open(GT_PROV_CSV, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    skip = 1 if first_line == "gt_province" else 0
    df = pd.read_csv(GT_PROV_CSV, skiprows=skip)
    print(f"\nIntegrating ground truth crops from gt_province.csv ({len(df)} rows)...")

    # Reproducible random split: 85% train, 15% valid
    np.random.seed(42)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="GT Province Crops"):
        img_rel = row.get("image")
        thai_name = row.get("gt_province")
        if pd.isna(img_rel) or pd.isna(thai_name):
            continue

        thai_name = str(thai_name).strip()
        if thai_name not in name_to_id:
            continue

        src_p = GT_PROV_DIR / img_rel
        if not src_p.exists():
            continue

        img = cv2.imread(str(src_p))
        if img is None or img.shape[0] < 5 or img.shape[1] < 5:
            continue

        prov_id = name_to_id[thai_name]
        split = "train" if np.random.rand() < 0.85 else "valid"

        folder_name = f"{prov_id:02d}_{thai_name}"
        target_folder = OUTPUT_DIR / split / folder_name
        target_folder.mkdir(parents=True, exist_ok=True)

        crop_filename = f"gt_{Path(img_rel).name}"
        crop_path = target_folder / crop_filename
        shutil.copy2(src_p, crop_path)

        metadata_rows.append({
            "filename": crop_filename,
            "rel_path": str(crop_path.relative_to(OUTPUT_DIR)),
            "split": split,
            "source": "ground_truth_crops",
            "province_id": prov_id,
            "thai_name": thai_name,
            "abbr": "",
            "width": img.shape[1],
            "height": img.shape[0],
        })


def main():
    print("=== Extracting Thai Province Dataset Crops ===")
    abbr_map, prov_map, name_to_id = load_mappings()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    # 1. Extract from Roboflow dataset
    extract_from_roboflow(abbr_map, metadata_rows)

    # 2. Integrate existing ground truth crops
    extract_from_gt_province(name_to_id, metadata_rows)

    # 3. Save metadata.csv
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = OUTPUT_DIR / "metadata.csv"
    meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")

    print(f"\nExtraction complete! Total crops: {len(meta_df)}")
    print(f"Saved metadata to: {meta_path}")
    print("\nCrops per split:")
    print(meta_df["split"].value_counts())
    print("\nUnique provinces covered:", meta_df["province_id"].nunique(), "/ 77")
    print("Top 10 provinces by count:")
    print(meta_df["thai_name"].value_counts().head(10))


if __name__ == "__main__":
    main()
