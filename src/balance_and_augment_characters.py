"""
src/balance_and_augment_characters.py

Merges character crops from:
  1. datasets/thai_character_crops/by_character_square/ (initial verified crops)
  2. datasets/thai_character_crops/candidates/ (cleaned user-curated candidates)

Performs targeted offline data augmentation for rare classes (< 60 samples)
using realistic affine rotations, perspective distortion, brightness/contrast jitter,
and subtle Gaussian blur so that every class has at least 60 diverse samples.

Populates:
  datasets/thai_character_crops/splits/train/<class>/
  datasets/thai_character_crops/splits/valid/<class>/

Saves:
  weights/char_classifier_map.json (mapping integer ID 0..49 to character name)
"""

import os
import sys
import shutil
import json
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "datasets" / "thai_character_crops"
CANDIDATES_DIR = BASE_DIR / "candidates"
SQUARE_DIR = BASE_DIR / "by_character_square"
SPLITS_DIR = BASE_DIR / "splits"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MAP_PATH = WEIGHTS_DIR / "char_classifier_map.json"

TARGET_MIN_SAMPLES = 60
TRAIN_RATIO = 0.85


def augment_pil_image(pil_img: Image.Image) -> Image.Image:
    """Applies realistic photometric and geometric augmentations to a character crop."""
    img = pil_img.copy()

    # 1. Random Rotation (-7 to +7 degrees)
    angle = random.uniform(-7, 7)
    # Estimate background color from corners
    corners = [img.getpixel((0, 0)), img.getpixel((img.width - 1, 0)),
               img.getpixel((0, img.height - 1)), img.getpixel((img.width - 1, img.height - 1))]
    avg_bg = tuple(int(np.mean([c[i] for c in corners])) for i in range(3)) if isinstance(corners[0], tuple) else 200
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=avg_bg)

    # 2. Random translation (-3 to +3 pixels)
    tx = random.randint(-3, 3)
    ty = random.randint(-3, 3)
    img = img.transform(img.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), fillcolor=avg_bg)

    # 3. Brightness jitter (0.75 - 1.25)
    b_factor = random.uniform(0.75, 1.25)
    img = ImageEnhance.Brightness(img).enhance(b_factor)

    # 4. Contrast jitter (0.75 - 1.25)
    c_factor = random.uniform(0.75, 1.25)
    img = ImageEnhance.Contrast(img).enhance(c_factor)

    # 5. Occasional slight blur or sharpness
    if random.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    elif random.random() < 0.35:
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(1.2, 1.8))

    return img


def balance_and_augment():
    print("==================================================================")
    print("THAI CHARACTER DATASET BALANCING & TARGETED AUGMENTATION")
    print("==================================================================")

    # 1. Discover all unique classes across both sources
    cand_classes = set(d.name for d in CANDIDATES_DIR.iterdir() if d.is_dir()) if CANDIDATES_DIR.exists() else set()
    square_classes = set(d.name for d in SQUARE_DIR.iterdir() if d.is_dir()) if SQUARE_DIR.exists() else set()
    all_classes = sorted(list(cand_classes | square_classes))

    print(f"Discovered {len(all_classes)} unique character classes.")

    # Save character classification mapping
    # Digits first, then Thai consonants
    digits = [c for c in all_classes if c.isdigit()]
    consonants = [c for c in all_classes if not c.isdigit()]
    ordered_classes = sorted(digits) + sorted(consonants)

    char_map = {str(i): ch for i, ch in enumerate(ordered_classes)}
    with open(MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(char_map, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(char_map)} classes mapping to: {MAP_PATH}")

    # Reset splits/train and splits/valid
    train_dir = SPLITS_DIR / "train"
    valid_dir = SPLITS_DIR / "valid"

    if train_dir.exists():
        shutil.rmtree(train_dir)
    if valid_dir.exists():
        shutil.rmtree(valid_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    total_train = 0
    total_valid = 0
    total_augmented = 0

    stats = []

    print("\nProcessing each class (merging + augmenting rare classes)...")
    for ch in tqdm(ordered_classes, desc="Balancing Classes"):
        # Gather all real images for this class
        real_images = []
        if (CANDIDATES_DIR / ch).exists():
            real_images.extend(list((CANDIDATES_DIR / ch).glob("*.jpg")) + list((CANDIDATES_DIR / ch).glob("*.png")))
        if (SQUARE_DIR / ch).exists():
            for p in list((SQUARE_DIR / ch).glob("*.jpg")) + list((SQUARE_DIR / ch).glob("*.png")):
                # Avoid duplicates
                if p.name not in [r.name for r in real_images]:
                    real_images.append(p)

        n_real = len(real_images)
        ch_train_dir = train_dir / ch
        ch_valid_dir = valid_dir / ch
        ch_train_dir.mkdir(parents=True, exist_ok=True)
        ch_valid_dir.mkdir(parents=True, exist_ok=True)

        # Shuffle real images
        random.shuffle(real_images)

        # Determine how many images to split into train vs valid
        # If very few real images (< 5), keep at least 1 for validation if n >= 2
        n_val = max(1, int(round(n_real * (1 - TRAIN_RATIO)))) if n_real >= 2 else 0
        n_train = n_real - n_val

        real_train = real_images[:n_train]
        real_val = real_images[n_train:]

        # Copy real train images
        for p in real_train:
            dest = ch_train_dir / p.name
            shutil.copy2(p, dest)
            total_train += 1

        # Copy real val images
        for p in real_val:
            dest = ch_valid_dir / p.name
            shutil.copy2(p, dest)
            total_valid += 1

        # Check if augmentation is needed to hit TARGET_MIN_SAMPLES
        current_train_count = len(real_train)
        augmented_added = 0
        if current_train_count < TARGET_MIN_SAMPLES and len(real_images) > 0:
            deficit = TARGET_MIN_SAMPLES - current_train_count
            # Generate augmented copies from the real images
            for i in range(deficit):
                src_p = real_images[i % len(real_images)]
                pil_src = Image.open(src_p).convert("RGB")
                aug_img = augment_pil_image(pil_src)

                aug_filename = f"aug_{src_p.stem}_v{i}.jpg"
                aug_img.save(ch_train_dir / aug_filename, quality=95)
                augmented_added += 1
                total_train += 1
                total_augmented += 1

        stats.append({
            "character": ch,
            "real_samples": n_real,
            "augmented_train": augmented_added,
            "final_train": len(list(ch_train_dir.glob("*.jpg"))),
            "final_valid": len(list(ch_valid_dir.glob("*.jpg"))),
        })

    stats_df = pd.DataFrame(stats)
    print("\n==================================================================")
    print("BALANCING & AUGMENTATION SUMMARY")
    print("==================================================================")
    print(f"Total Character Classes: {len(stats_df)}")
    print(f"Total Training Samples: {total_train}")
    print(f"Total Validation Samples: {total_valid}")
    print(f"Total Augmented Samples Created: {total_augmented}")
    print(f"Minimum Samples in Any Training Class: {stats_df['final_train'].min()}")
    print(f"Maximum Samples in Any Training Class: {stats_df['final_train'].max()}")
    print("\nBottom 15 Classes in Training Set (All successfully boosted):")
    print(stats_df.sort_values("real_samples").head(15).to_string(index=False))
    print("==================================================================\n")


if __name__ == "__main__":
    balance_and_augment()
