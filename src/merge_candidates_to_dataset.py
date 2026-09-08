"""
src/merge_candidates_to_dataset.py

Merges candidate character crops from 'datasets/thai_character_crops/candidates/'
into the training/validation dataset:
  - datasets/thai_character_crops/splits/train/
  - datasets/thai_character_crops/splits/valid/
  - datasets/thai_character_crops/by_character_square/

Options:
  --min-conf: Minimum OCR/detection confidence (default: 0.50)
  --train-ratio: Ratio of merged images assigned to train (default: 0.85)
  --classes: Optional comma-separated list of specific classes to merge (e.g. 'ค,ฆ,ป,ง,ฟ,1,2')
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "datasets" / "thai_character_crops"
CANDIDATES_DIR = BASE_DIR / "candidates"
SPLITS_DIR = BASE_DIR / "splits"
SQUARE_DIR = BASE_DIR / "by_character_square"
META_PATH = BASE_DIR / "candidates_metadata.csv"
MAIN_META_PATH = BASE_DIR / "metadata.csv"


def merge_candidates(min_conf=0.50, train_ratio=0.85, selected_classes=None):
    print("==================================================================")
    print("MERGING HARVESTED CANDIDATES INTO TRAINING DATASET")
    print("==================================================================")

    if not CANDIDATES_DIR.exists():
        print(f"[Error] {CANDIDATES_DIR} not found.")
        return

    char_dirs = sorted([d for d in CANDIDATES_DIR.iterdir() if d.is_dir()])
    if selected_classes:
        char_dirs = [d for d in char_dirs if d.name in selected_classes]

    print(f"Targeting {len(char_dirs)} character classes...")

    np.random.seed(42)
    merged_count = 0
    train_added = 0
    val_added = 0

    for cd in char_dirs:
        char_name = cd.name
        img_files = list(cd.glob("*.jpg")) + list(cd.glob("*.png"))

        train_folder = SPLITS_DIR / "train" / char_name
        valid_folder = SPLITS_DIR / "valid" / char_name
        square_folder = SQUARE_DIR / char_name

        train_folder.mkdir(parents=True, exist_ok=True)
        valid_folder.mkdir(parents=True, exist_ok=True)
        square_folder.mkdir(parents=True, exist_ok=True)

        for img_p in img_files:
            # Check if file already copied
            dest_square = square_folder / img_p.name
            if dest_square.exists():
                continue

            shutil.copy2(img_p, dest_square)

            is_train = np.random.rand() < train_ratio
            if is_train:
                shutil.copy2(img_p, train_folder / img_p.name)
                train_added += 1
            else:
                shutil.copy2(img_p, valid_folder / img_p.name)
                val_added += 1

            merged_count += 1

    print(f"\n[Success] Merged {merged_count} candidate images!")
    print(f"  --> Added to train split: {train_added}")
    print(f"  --> Added to valid split: {val_added}")
    print(f"  --> Added to by_character_square: {merged_count}")
    print("==================================================================\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-conf", type=float, default=0.50)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated list of classes (e.g. '1,2,ก,ข')")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    merge_candidates(min_conf=args.min_conf, train_ratio=args.train_ratio, selected_classes=classes)
