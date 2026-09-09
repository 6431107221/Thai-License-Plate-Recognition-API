"""
src/balance_and_merge_lao_provinces.py

Builds a balanced, high-quality Lao province dataset by:
1. Ingesting user-cleaned, verified real crops from datasets/Lao/lao_province_candidates/.
2. Capping Class 00 (ນະຄອນຫຼວງວຽງຈັນ / ກຳແພງນະຄອນ) to 150 train, 25 val, 25 test samples.
3. Ingesting minority real crops (Luang Prabang, Sayaboury, Savannakhet, Champasak, etc.).
4. Generating photorealistic synthetic Lao banner crops using genuine macOS Lao fonts
   (Lao Sangam MN, Lao MN) with yellow, white, and blue plate themes, gradient lighting,
   sensor noise, affine skew, and Gaussian blur to fill empty & minority classes.
5. Guaranteeing exactly 150 train, 25 val, and 25 test crops per province (18 classes total).
"""

import os
import shutil
import random
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao_province_candidates"
CROPS_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao_province_crops"
BACKUP_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao_province_crops_backup"
MAP_PATH = PROJECT_ROOT / "weights" / "province_map_lao.json"

LAO_FONTS = [
    "/System/Library/Fonts/Supplemental/Lao Sangam MN.ttf",
    "/System/Library/Fonts/Supplemental/Lao MN.ttc"
]

TARGET_TRAIN = 150
TARGET_VAL = 25
TARGET_TEST = 25


def generate_synthetic_lao_banner(text: str, w: int = 256, h: int = 64) -> np.ndarray:
    """Renders an authentic, photorealistic Lao license plate province banner."""
    # Background theme: 75% yellow (private), 20% white (business/foreign), 5% blue (official)
    bg_type = random.choices(["yellow", "white", "blue"], weights=[0.75, 0.20, 0.05])[0]

    if bg_type == "yellow":
        r = random.randint(215, 245)
        g = random.randint(165, 195)
        b = random.randint(20, 45)
        bg_col = (r, g, b)
        text_col = (random.randint(15, 35), random.randint(15, 35), random.randint(15, 35))
    elif bg_type == "white":
        v = random.randint(220, 245)
        bg_col = (v, v, v)
        text_col = (random.randint(15, 35), random.randint(15, 35), random.randint(15, 35))
    else:  # blue
        bg_col = (random.randint(15, 35), random.randint(60, 95), random.randint(140, 185))
        text_col = (random.randint(230, 255), random.randint(230, 255), random.randint(230, 255))

    im = Image.new("RGB", (w, h), color=bg_col)

    # Gradient / non-uniform lighting across banner
    grad = np.linspace(random.uniform(0.85, 1.0), random.uniform(0.95, 1.15), w)
    im_np = np.array(im, dtype=np.float32)
    im_np = np.clip(im_np * grad[None, :, None], 0, 255).astype(np.uint8)
    im = Image.fromarray(im_np)
    draw = ImageDraw.Draw(im)

    # Font selection & sizing
    font_path = random.choice(LAO_FONTS)
    fsize = random.randint(28, 38)
    try:
        font = ImageFont.truetype(font_path, fsize)
    except Exception:
        font = ImageFont.truetype(LAO_FONTS[0], fsize)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # If text is too wide for banner, shrink font
    if tw > w - 20:
        fsize = int(fsize * (w - 20) / tw)
        font = ImageFont.truetype(font_path, max(20, fsize))
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x = (w - tw) // 2 + random.randint(-4, 4)
    y = (h - th) // 2 - 4 + random.randint(-2, 2)
    draw.text((x, y), text, fill=text_col, font=font)

    # Camera degradations
    img_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    # 1. Perspective / affine jitter
    angle = random.uniform(-3.5, 3.5)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_cv = cv2.warpAffine(img_cv, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Gaussian blur
    if random.random() < 0.7:
        ksize = random.choice([3, 5])
        img_cv = cv2.GaussianBlur(img_cv, (ksize, ksize), random.uniform(0.4, 1.2))

    # 3. Sensor noise
    noise = np.random.normal(0, random.uniform(3, 8), img_cv.shape).astype(np.float32)
    img_cv = np.clip(img_cv.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Optional slight JPEG compression artifact
    if random.random() < 0.5:
        quality = random.randint(55, 85)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode(".jpg", img_cv, encode_param)
        img_cv = cv2.imdecode(enc, 1)

    return img_cv


def balance_and_merge():
    print("=======================================================")
    print("--- Lao Province Dataset Balancing & Merging Pipeline ---")
    print(f"Candidates Source : {CANDIDATES_DIR}")
    print(f"Target Crops Dir  : {CROPS_DIR}")
    print(f"Target per Class  : Train={TARGET_TRAIN}, Val={TARGET_VAL}, Test={TARGET_TEST}")
    print("=======================================================\n")

    random.seed(42)
    np.random.seed(42)

    with open(MAP_PATH, "r", encoding="utf-8") as f:
        prov_map = json.load(f)

    # Backup existing crops folder if not backed up
    if CROPS_DIR.exists() and not BACKUP_DIR.exists():
        print(f"Backing up existing crops to {BACKUP_DIR}...")
        shutil.copytree(CROPS_DIR, BACKUP_DIR)
        print("Backup complete.")

    # Recreate target crops dir
    if CROPS_DIR.exists():
        shutil.rmtree(CROPS_DIR)

    for split in ["train", "valid", "test"]:
        for pid_str, pname in prov_map.items():
            folder_name = f"{int(pid_str):02d}_{pname}"
            (CROPS_DIR / split / folder_name).mkdir(parents=True, exist_ok=True)

    summary_stats = []

    for pid in range(len(prov_map)):
        pname = prov_map[str(pid)]
        folder_name = f"{pid:02d}_{pname}"
        cand_folder = CANDIDATES_DIR / folder_name

        # 1. Collect valid real candidate crops
        real_files = []
        if cand_folder.exists():
            for p in cand_folder.glob("*.jpg"):
                if p.stat().st_size > 200:
                    real_files.append(p)

        random.shuffle(real_files)
        n_real = len(real_files)

        # Allocate real images across splits
        # Class 00 has ~940 images; we cap real images to 150 train, 25 val, 25 test
        if pid == 0:
            real_train = real_files[:TARGET_TRAIN]
            real_val = real_files[TARGET_TRAIN:TARGET_TRAIN + TARGET_VAL]
            real_test = real_files[TARGET_TRAIN + TARGET_VAL:TARGET_TRAIN + TARGET_VAL + TARGET_TEST]
        else:
            # Minority classes: split real images proportionally (70% train, 15% val, 15% test)
            n_test = min(TARGET_TEST, int(n_real * 0.15))
            n_val = min(TARGET_VAL, int(n_real * 0.15))
            n_train = n_real - n_val - n_test

            real_test = real_files[:n_test]
            real_val = real_files[n_test:n_test + n_val]
            real_train = real_files[n_test + n_val:]

        # Copy real crops
        counts = {"train": {"real": 0, "synth": 0}, "valid": {"real": 0, "synth": 0}, "test": {"real": 0, "synth": 0}}

        for split, flist, target_n in [
            ("train", real_train, TARGET_TRAIN),
            ("valid", real_val, TARGET_VAL),
            ("test", real_test, TARGET_TEST),
        ]:
            out_class_dir = CROPS_DIR / split / folder_name

            # Copy real crops
            for idx, src_p in enumerate(flist):
                dest_p = out_class_dir / f"real_{idx:04d}_{src_p.name}"
                shutil.copy2(src_p, dest_p)
                counts[split]["real"] += 1

            # Fill remainder with photorealistic synthetic Lao crops
            needed_synth = target_n - counts[split]["real"]
            for s_idx in range(needed_synth):
                # For Class 00, occasionally render both older Kamphaeng Nakhon and Nakhon Luang
                if pid == 0:
                    banner_text = random.choice(["ນະຄອນຫຼວງວຽງຈັນ", "ກຳແພງນະຄອນ"])
                else:
                    banner_text = pname

                synth_im = generate_synthetic_lao_banner(banner_text)
                dest_p = out_class_dir / f"synth_{s_idx:04d}_{banner_text}.jpg"
                cv2.imwrite(str(dest_p), synth_im)
                counts[split]["synth"] += 1

        summary_stats.append({
            "id": pid,
            "name": pname,
            "real_total": n_real,
            "train_real": counts["train"]["real"],
            "train_synth": counts["train"]["synth"],
            "val_real": counts["valid"]["real"],
            "val_synth": counts["valid"]["synth"],
            "test_real": counts["test"]["real"],
            "test_synth": counts["test"]["synth"],
        })

    # Print distribution summary table
    print("\nBalanced Lao Province Dataset Summary:")
    print(f"{'ID':<4} {'Province':<22} {'Real Total':<12} {'Train (R+S)':<15} {'Val (R+S)':<12} {'Test (R+S)':<12}")
    print("-" * 80)
    for st in summary_stats:
        train_str = f"{st['train_real']}R + {st['train_synth']}S"
        val_str = f"{st['val_real']}R + {st['val_synth']}S"
        test_str = f"{st['test_real']}R + {st['test_synth']}S"
        print(f"[{st['id']:02d}] {st['name']:<22} {st['real_total']:<12} {train_str:<15} {val_str:<12} {test_str:<12}")

    print("-" * 80)
    print(f"Total Train images : {len(prov_map) * TARGET_TRAIN}")
    print(f"Total Valid images : {len(prov_map) * TARGET_VAL}")
    print(f"Total Test images  : {len(prov_map) * TARGET_TEST}")
    print("\nDataset ready for training!")


if __name__ == "__main__":
    balance_and_merge()
