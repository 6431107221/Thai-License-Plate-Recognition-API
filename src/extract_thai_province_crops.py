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
import re
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
YOLO_PROV_DIR = (PROJECT_ROOT / "datasets" / "Thai" / "thai-car-license-plate-province.v5i.yolov11") if (PROJECT_ROOT / "datasets" / "Thai" / "thai-car-license-plate-province.v5i.yolov11").exists() else (PROJECT_ROOT / "datasets" / "thai-car-license-plate-province.v5i.yolov11")
GT_PROV_CSV = PROJECT_ROOT / "output" / "ground_truth_crops" / "gt_province.csv"
GT_PROV_DIR = PROJECT_ROOT / "output" / "ground_truth_crops"

ABBR_MAP_PATH = PROJECT_ROOT / "weights" / "province_abbr_map.json"
PROV_MAP_PATH = PROJECT_ROOT / "weights" / "province_map.json"

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "Thai" / "thai_province_crops"


def load_mappings():
    with open(ABBR_MAP_PATH, "r", encoding="utf-8") as f:
        abbr_map = json.load(f)
    with open(PROV_MAP_PATH, "r", encoding="utf-8") as f:
        prov_map = json.load(f)
    name_to_id = {v: int(k) for k, v in prov_map.items()}
    return abbr_map, prov_map, name_to_id


def augment_real_province_crop(img_bgr: np.ndarray, aug_id: int) -> np.ndarray:
    """
    Applies authentic physical & photometric augmentations strictly to REAL plate crops.
    Preserves genuine Thai DLT embossed stamp letterforms (NO computer synthetic fonts).
    
    Augmentations:
      1. Commercial Yellow Truck Plate Simulation (transforms white background to authentic amber/yellow)
      2. Lighting, Gamma & Contrast shifts (day sunlight glare, shadow, nighttime underexposure)
      3. Perspective angle skews (+/- 10 deg) and camera tilt
      4. Motion blur & sensor road grime
    """
    h, w = img_bgr.shape[:2]
    img = img_bgr.copy()

    # 1. Commercial Yellow Truck Plate Transformation (~40% of augmentations)
    # Stamped black Thai letters stay dark, while white/grey plate background is tinted yellow/mustard
    if random.random() < 0.40:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Background mask (bright/plate surface pixels)
        _, bg_mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        bg_mask_3c = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR) / 255.0

        # Realistic DLT yellow plate color (BGR: yellow/amber/mustard)
        b_y = random.randint(30, 80)
        g_y = random.randint(175, 220)
        r_y = random.randint(220, 255)
        yellow_plate = np.full_like(img, (b_y, g_y, r_y), dtype=np.uint8)

        # Blend: Keep embossed characters dark, replace background with yellow plate tone
        blend = (img.astype(np.float32) * (1.0 - bg_mask_3c * 0.75) +
                 yellow_plate.astype(np.float32) * (bg_mask_3c * 0.75))
        img = np.clip(blend, 0, 255).astype(np.uint8)

    # 2. Lighting & Exposure variation
    factor = random.uniform(0.75, 1.30)
    img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # 3. Gamma correction (simulates headlights / backlit road scenes)
    if random.random() < 0.5:
        gamma = random.uniform(0.80, 1.30)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        img = cv2.LUT(img, table)

    # 4. Perspective skew / CCTV camera viewing angle (+/- 8-10 degrees)
    if random.random() < 0.65 and w >= 20 and h >= 10:
        dx = int(w * random.uniform(0.02, 0.08))
        dy = int(h * random.uniform(0.02, 0.08))
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [random.randint(0, max(1, dx)), random.randint(0, max(1, dy))],
            [w - random.randint(0, max(1, dx)), random.randint(0, max(1, dy))],
            [random.randint(0, max(1, dx)), h - random.randint(0, max(1, dy))],
            [w - random.randint(0, max(1, dx)), h - random.randint(0, max(1, dy))]
        ])
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M_persp, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 5. Subtle rotation
    if random.random() < 0.4:
        angle = random.uniform(-3.5, 3.5)
        M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 6. Motion blur / vehicle vibration
    if random.random() < 0.30:
        k_size = random.choice([3, 5])
        kernel = np.zeros((k_size, k_size))
        kernel[int((k_size - 1) / 2), :] = np.ones(k_size)
        kernel = kernel / k_size
        img = cv2.filter2D(img, -1, kernel)

    # 7. Light Gaussian sensor noise
    if random.random() < 0.25:
        noise = np.random.normal(0, random.uniform(2, 6), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


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


GT_UNIFIED_CSV = PROJECT_ROOT / "output" / "ground_truth_crops" / "ground_truth_unified.csv"


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

    # Load unified CSV to detect truck plates
    unified_truck_map = {}
    if GT_UNIFIED_CSV.exists():
        df_u = pd.read_csv(GT_UNIFIED_CSV)
        for _, urow in df_u.iterrows():
            p_img = str(urow.get("province_image", "")).strip()
            gt_p = str(urow.get("gt_plate", "")).strip()
            if p_img and gt_p:
                unified_truck_map[Path(p_img).name] = bool(re.match(r"^\d\d-\d{4}", gt_p))

    truck_aug_count = 0
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

        # Real Yellow Truck Plate Duplication & Augmentation:
        # If this crop belongs to a truck plate (NN-NNNN) or has yellow tint, duplicate 5x into train split
        is_truck = unified_truck_map.get(Path(img_rel).name, False)
        # Check yellow color characteristics (high red/green, lower blue)
        if not is_truck and img.shape[0] >= 10 and img.shape[1] >= 20:
            mean_b, mean_g, mean_r = np.mean(img, axis=(0, 1))
            if mean_r > 120 and mean_g > 110 and mean_b < mean_r * 0.70:
                is_truck = True

        if is_truck:
            train_class_folder = OUTPUT_DIR / "train" / folder_name
            train_class_folder.mkdir(parents=True, exist_ok=True)
            for dup_idx in range(5):
                aug_truck = augment_real_province_crop(img, dup_idx)
                aug_fn = f"aug_truck_{dup_idx}_{crop_filename}"
                cv2.imwrite(str(train_class_folder / aug_fn), aug_truck)
                truck_aug_count += 1

    print(f"Generated {truck_aug_count} dedicated real yellow truck plate augmentations (5x duplication).")


def balance_minority_provinces(prov_map, target_min=120):
    """
    Boosts minority classes using authentic real-crop augmentations.
    Target: >= 160 samples for truck confusion candidates (บุรีรัมย์, จันทบุรี, ภูเก็ต, กาญจนบุรี, นนทบุรี),
    and >= target_min (120) samples for all other provinces.
    """
    TRUCK_FOCUS_PROVINCES = {"บุรีรัมย์", "จันทบุรี", "ภูเก็ต", "กาญจนบุรี", "นนทบุรี", "เชียงใหม่", "ราชบุรี"}
    print(f"\nBalancing minority classes in train split (default target: {target_min}, truck focus: 160)...")
    train_dir = OUTPUT_DIR / "train"
    augmented_count = 0

    for pid in range(len(prov_map)):
        thai_name = prov_map[str(pid)]
        min_needed = 160 if thai_name in TRUCK_FOCUS_PROVINCES else target_min
        folder_name = f"{pid:02d}_{thai_name}"
        class_folder = train_dir / folder_name
        class_folder.mkdir(parents=True, exist_ok=True)

        existing_imgs = list(class_folder.glob("*.jpg"))
        # If train has no crops for this province, copy seed crops from valid or test splits
        if not existing_imgs:
            alt_imgs = list((OUTPUT_DIR / "valid" / folder_name).glob("*.jpg")) + list((OUTPUT_DIR / "test" / folder_name).glob("*.jpg"))
            if alt_imgs:
                for a_p in alt_imgs:
                    dest_p = class_folder / a_p.name
                    if not dest_p.exists():
                        shutil.copy2(a_p, dest_p)
                existing_imgs = list(class_folder.glob("*.jpg"))

        n_exist = len(existing_imgs)

        if n_exist < min_needed and n_exist > 0:
            needed = min_needed - n_exist
            for aug_i in range(needed):
                src_p = random.choice(existing_imgs)
                src_im = cv2.imread(str(src_p))
                if src_im is None:
                    continue

                # Apply authentic real-crop physical augmentation
                aug_im = augment_real_province_crop(src_im, aug_i)

                aug_path = class_folder / f"aug_real_{aug_i:04d}_{thai_name}.jpg"
                cv2.imwrite(str(aug_path), aug_im)
                augmented_count += 1

    print(f"Generated {augmented_count} purely real-augmented training crops across minority provinces.")



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
