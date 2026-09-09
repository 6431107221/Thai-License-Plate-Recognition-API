"""
src/extract_thai_province_crops.py

Extracts Thai license plate province crops from:
  1. datasets/thai-car-license-plate-province.v5i.yolov11 (3,217 rectangular plates, skipping 4 polygon files)
  2. output/ground_truth_crops/gt_province.csv (798 crops)

Features:
  - Strict bounding-box validation (skips non-rectangular polygons cleanly)
  - Balanced minority class augmentation & synthetic rendering (boosts all 77 classes to >= 100 train samples)
  - Standard ImageFolder splits:
      datasets/thai_province_crops/
        train/<province_id>_<thai_name>/
        valid/<province_id>_<thai_name>/
        test/<province_id>_<thai_name>/
"""

import os
import json
import yaml
import shutil
import random
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_PROV_DIR = PROJECT_ROOT / "datasets" / "thai-car-license-plate-province.v5i.yolov11"
GT_PROV_CSV = PROJECT_ROOT / "output" / "ground_truth_crops" / "gt_province.csv"
GT_PROV_DIR = PROJECT_ROOT / "output" / "ground_truth_crops"

ABBR_MAP_PATH = PROJECT_ROOT / "weights" / "province_abbr_map.json"
PROV_MAP_PATH = PROJECT_ROOT / "weights" / "province_map.json"
FONT_PATH = PROJECT_ROOT / "src" / "scratch" / "fonts" / "Sarabun-Bold.ttf"

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "thai_province_crops"


def load_mappings():
    with open(ABBR_MAP_PATH, "r", encoding="utf-8") as f:
        abbr_map = json.load(f)
    with open(PROV_MAP_PATH, "r", encoding="utf-8") as f:
        prov_map = json.load(f)
    name_to_id = {v: int(k) for k, v in prov_map.items()}
    return abbr_map, prov_map, name_to_id


def augment_crop(img_bgr: np.ndarray, aug_id: int) -> np.ndarray:
    """Applies realistic geometric & photometric augmentations to province crops."""
    h, w = img_bgr.shape[:2]
    img = img_bgr.copy()

    # 1. Subtle lighting
    factor = random.uniform(0.80, 1.20)
    img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # 2. Contrast
    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.25)
        mean_v = np.mean(img)
        img = np.clip((img.astype(np.float32) - mean_v) * alpha + mean_v, 0, 255).astype(np.uint8)

    # 3. Slight tilt (keep letters clearly legible)
    if random.random() < 0.6:
        angle = random.uniform(-2.5, 2.5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 4. Subtle blur
    if random.random() < 0.35:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def generate_synthetic_province(thai_name: str, font_path: Path, width=200, height=50) -> np.ndarray:
    """Renders authentic Thai province text banner on realistic plate background (white passenger & yellow commercial)."""
    # 50% white passenger plate, 50% yellow commercial truck plate
    is_yellow_truck = random.random() < 0.50
    if is_yellow_truck:
        # Commercial yellow truck plate background (BGR: yellow/mustard/amber)
        b = random.randint(35, 95)
        g = random.randint(170, 225)
        r = random.randint(215, 255)
        bg = np.full((height, width, 3), (b, g, r), dtype=np.uint8)
    else:
        bg_val = random.randint(215, 245)
        bg = np.full((height, width, 3), (bg_val, bg_val, bg_val), dtype=np.uint8)

    noise = np.random.normal(0, 4, (height, width, 3)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    pil_im = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)
    font_size = random.randint(22, 26)
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), thai_name, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = max(4, (width - tw) // 2)
    ty = max(2, (height - th) // 2 - 2)
    text_val = random.randint(15, 40)
    draw.text((tx, ty), thai_name, fill=(text_val, text_val, text_val), font=font)

    im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    if random.random() < 0.4:
        im = cv2.GaussianBlur(im, (3, 3), 0)
    return im


def extract_from_roboflow(abbr_map, metadata_rows):
    data_yaml_path = YOLO_PROV_DIR / "data.yaml"
    if not data_yaml_path.exists():
        print(f"[Warning] {data_yaml_path} not found.")
        return

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        ydata = yaml.safe_load(f)
    class_names = ydata["names"]

    skipped_polygons = 0
    extracted_bboxes = 0

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
                # STRICT RECTANGULAR CHECK: Skip polygons with > 5 coordinates
                if len(parts) != 5:
                    skipped_polygons += 1
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

                if (x2 - x1) < 8 or (y2 - y1) < 6:
                    continue

                crop = img[y1:y2, x1:x2]
                folder_name = f"{prov_id:02d}_{thai_name}"
                target_folder = OUTPUT_DIR / split / folder_name
                target_folder.mkdir(parents=True, exist_ok=True)

                crop_filename = f"rf_{split}_{stem}_b{box_idx}.jpg"
                crop_path = target_folder / crop_filename
                cv2.imwrite(str(crop_path), crop)
                extracted_bboxes += 1

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

    print(f"Extracted {extracted_bboxes} clean bboxes (Skipped {skipped_polygons} non-rectangular polygons)")


def extract_from_gt_province(name_to_id, metadata_rows):
    if not GT_PROV_CSV.exists():
        print(f"[Warning] {GT_PROV_CSV} not found.")
        return

    with open(GT_PROV_CSV, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    skip = 1 if first_line == "gt_province" else 0
    df = pd.read_csv(GT_PROV_CSV, skiprows=skip)
    print(f"\nIntegrating ground truth crops from gt_province.csv ({len(df)} rows)...")

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


def balance_minority_provinces(prov_map, target_min=100):
    """Boosts any province with < target_min training samples using augmentation & synthetic rendering."""
    print(f"\nBalancing minority classes in train split (target minimum: {target_min} images/province)...")
    train_dir = OUTPUT_DIR / "train"
    augmented_count = 0

    for pid in range(len(prov_map)):
        thai_name = prov_map[str(pid)]
        folder_name = f"{pid:02d}_{thai_name}"
        class_folder = train_dir / folder_name
        class_folder.mkdir(parents=True, exist_ok=True)

        existing_imgs = list(class_folder.glob("*.jpg"))
        n_exist = len(existing_imgs)

        if n_exist < target_min:
            needed = target_min - n_exist
            for aug_i in range(needed):
                if n_exist > 0 and (aug_i % 2 == 0 or not FONT_PATH.exists()):
                    # Augment existing crop
                    src_p = random.choice(existing_imgs)
                    src_im = cv2.imread(str(src_p))
                    if src_im is not None:
                        aug_im = augment_crop(src_im, aug_i)
                    else:
                        aug_im = generate_synthetic_province(thai_name, FONT_PATH)
                else:
                    # Synthetic render with authentic font
                    aug_im = generate_synthetic_province(thai_name, FONT_PATH)

                aug_path = class_folder / f"aug_{aug_i:04d}_{thai_name}.jpg"
                cv2.imwrite(str(aug_path), aug_im)
                augmented_count += 1

    print(f"Generated {augmented_count} balanced training crops across minority provinces.")


def main():
    print("=== Extracting & Balancing Thai Province Dataset Crops ===")
    abbr_map, prov_map, name_to_id = load_mappings()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    # 1. Extract clean rectangular bounding boxes from Roboflow dataset
    extract_from_roboflow(abbr_map, metadata_rows)

    # 2. Integrate existing ground truth crops
    extract_from_gt_province(name_to_id, metadata_rows)

    # 3. Balance all 77 provinces to >= 100 training samples
    balance_minority_provinces(prov_map, target_min=100)

    # 4. Save metadata.csv
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = OUTPUT_DIR / "metadata.csv"
    meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")

    print(f"\nExtraction complete! Real crops recorded: {len(meta_df)}")
    print(f"Saved metadata to: {meta_path}")

    # Summary of final training counts
    train_dir = OUTPUT_DIR / "train"
    final_counts = [len(list((train_dir / f"{pid:02d}_{prov_map[str(pid)]}").glob("*.jpg"))) for pid in range(len(prov_map))]
    print(f"\nFinal Train Split Balance:")
    print(f"  Min samples in any province: {min(final_counts)}")
    print(f"  Max samples in any province: {max(final_counts)}")
    print(f"  Total train images: {sum(final_counts)}")
    print(f"  All 77 provinces >= 100 samples: {all(c >= 100 for c in final_counts)}")


if __name__ == "__main__":
    main()
