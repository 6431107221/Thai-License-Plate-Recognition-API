"""
src/extract_lao_province_crops.py

Extracts isolated province banner crops from all 11,317 Lao license plate images
using the user's vertical flip + Thai Model 2 (component_detector.pt) detection workflow.

Workflow:
1. For each Lao plate image:
   - Flip vertically: cv2.flip(im, 0)
   - Run Model 2 component detector to find the province box
   - Remap coordinates back to upright orientation: y_orig = H - y_flipped
   - Fallback if low confidence: slice top banner [2%:38%, 15%:85%]
2. Saves crops into:
   datasets/lao_province_crops/{train, valid, test}/{province_id:02d}_{province_name}/
3. Balances minority classes in train split by augmenting samples up to 150 images per class.
"""

import os
import sys
import json
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAO_DATASET_DIR = PROJECT_ROOT / "datasets" / "lao-plate-dataset"
IMAGES_DIR = LAO_DATASET_DIR / "images"
CSV_PATH = LAO_DATASET_DIR / "ground_truth_all.csv"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "lao_province_crops"
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "component_detector.pt"
MAP_PATH = PROJECT_ROOT / "weights" / "province_map_lao.json"


def augment_crop(img_bgr: np.ndarray, aug_id: int) -> np.ndarray:
    """Applies realistic augmentation to province banner crops."""
    h, w = img_bgr.shape[:2]
    img = img_bgr.copy()

    # 1. Subtle lighting / brightness
    factor = random.uniform(0.75, 1.25)
    img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # 2. Contrast
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)
        mean_v = np.mean(img)
        img = np.clip((img.astype(np.float32) - mean_v) * alpha + mean_v, 0, 255).astype(np.uint8)

    # 3. Slight rotation / affine (keep text readable)
    if random.random() < 0.6:
        angle = random.uniform(-3.0, 3.0)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 4. Subtle blur / sharpness
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def extract_lao_provinces(target_min_train_samples=150):
    print(f"--- Extracting Lao Province Crops using Flip-and-Detect Workflow ---")
    if not CSV_PATH.exists() or not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Lao dataset not found at {LAO_DATASET_DIR}")

    with open(MAP_PATH, "r", encoding="utf-8") as f:
        prov_map = json.load(f)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} records from {CSV_PATH.name}")

    print(f"Loading Thai Model 2 from {WEIGHTS_PATH}...")
    model2 = YOLO(str(WEIGHTS_PATH))

    # Prepare directories
    for split in ["train", "valid", "test"]:
        for pid_str, pname in prov_map.items():
            folder_name = f"{int(pid_str):02d}_{pname}"
            (OUTPUT_DIR / split / folder_name).mkdir(parents=True, exist_ok=True)

    extracted_counts = {"train": {}, "valid": {}, "test": {}}
    for s in extracted_counts:
        for pid in range(len(prov_map)):
            extracted_counts[s][pid] = 0

    model2_detected = 0
    fallback_used = 0

    # Batch process
    print("Extracting crops with Model 2...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        fn = row["filename"]
        img_p = IMAGES_DIR / fn
        if not img_p.exists() or img_p.stat().st_size < 300:
            continue

        raw_split = str(row["split"]).strip().lower()
        split = "valid" if "val" in raw_split else ("test" if "test" in raw_split else "train")
        pid = int(row["province_id"])
        pname = prov_map.get(str(pid), f"prov_{pid}")
        folder_name = f"{pid:02d}_{pname}"

        im = cv2.imread(str(img_p))
        if im is None or im.size == 0:
            continue

        rh, rw = im.shape[:2]

        # --- USER'S WORKFLOW: Vertical Flip -> Model 2 -> Remap ---
        flipped = cv2.flip(im, 0)
        try:
            res = model2(flipped, conf=0.20, verbose=False)[0]
        except Exception:
            res = None

        prov_crop = None
        best_conf = 0.0

        if res is not None and len(res.boxes) > 0:
            for b in res.boxes:
                c_name = model2.names[int(b.cls[0])].lower()
                c_conf = float(b.conf[0])
                if "prov" in c_name and c_conf > best_conf:
                    bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().astype(int)
                    # Remap from flipped: y_orig = rh - y_flipped
                    orig_y1 = max(0, rh - by2)
                    orig_y2 = min(rh, rh - by1)
                    orig_x1 = max(0, bx1)
                    orig_x2 = min(rw, bx2)

                    crop_candidate = im[orig_y1:orig_y2, orig_x1:orig_x2]
                    if crop_candidate.shape[0] >= 6 and crop_candidate.shape[1] >= 15:
                        prov_crop = crop_candidate
                        best_conf = c_conf

        if prov_crop is not None:
            model2_detected += 1
        else:
            # Fallback: slice top banner (provinces are at the top 35% of Lao plates)
            fallback_used += 1
            y1 = int(rh * 0.03)
            y2 = int(rh * 0.36)
            x1 = int(rw * 0.12)
            x2 = int(rw * 0.88)
            prov_crop = im[y1:y2, x1:x2]

        if prov_crop is None or prov_crop.size == 0:
            continue

        # Save crop
        stem = Path(fn).stem
        save_path = OUTPUT_DIR / split / folder_name / f"{stem}_prov.jpg"
        cv2.imwrite(str(save_path), prov_crop)
        extracted_counts[split][pid] += 1

    print(f"\nCrop Extraction Complete:")
    print(f"  Model 2 Detections : {model2_detected} ({model2_detected/(model2_detected+fallback_used)*100:.1f}%)")
    print(f"  Fallback Crops     : {fallback_used} ({fallback_used/(model2_detected+fallback_used)*100:.1f}%)")

    # --- Minority Class Balancing / Augmentation in Train Split ---
    print(f"\nBalancing minority classes in train split (target min: {target_min_train_samples} samples)...")
    augmented_count = 0
    train_dir = OUTPUT_DIR / "train"

    for pid in range(len(prov_map)):
        pname = prov_map[str(pid)]
        folder_name = f"{pid:02d}_{pname}"
        class_folder = train_dir / folder_name
        existing_imgs = list(class_folder.glob("*.jpg"))
        n_exist = len(existing_imgs)

        if n_exist == 0:
            continue

        if n_exist < target_min_train_samples:
            needed = target_min_train_samples - n_exist
            print(f"  Class {pid:02d} ({pname}): {n_exist} images -> augmenting +{needed} images")
            
            aug_idx = 0
            while aug_idx < needed:
                src_p = random.choice(existing_imgs)
                src_im = cv2.imread(str(src_p))
                if src_im is None:
                    continue
                aug_im = augment_crop(src_im, aug_idx)
                aug_save_p = class_folder / f"aug_{aug_idx:04d}_{src_p.stem}.jpg"
                cv2.imwrite(str(aug_save_p), aug_im)
                aug_idx += 1
                augmented_count += 1

    print(f"\nTotal synthetic augmented crops generated: {augmented_count}")

    # Summary
    print("\nFinal Class Distribution in Train Split:")
    for pid in range(len(prov_map)):
        pname = prov_map[str(pid)]
        folder_name = f"{pid:02d}_{pname}"
        total_train = len(list((train_dir / folder_name).glob("*.jpg")))
        print(f"  [{pid:02d}] {pname:<20}: {total_train} images")


if __name__ == "__main__":
    extract_lao_provinces()
