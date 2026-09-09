"""
src/train_unified_plate_detector.py

Trains a high-recall, unified YOLO11 plate polygon segmentation detector
on both Thai and Lao vehicle images.

Dataset:
  datasets/unified_plate_polygon/data.yaml
  - Thai vehicles: LPR 2 - Polygon (1,068 train images)
  - Lao vehicles: laos plate v3 + Lao License Plates v2 (1,749 train images)
  Total: ~2,800 train vehicle images

Saves:
  weights/plate_polygon_detector.pt (Unified Thai + Lao Detector)
  weights/plate_polygon_detector_thai_only.pt (Backup of previous model)
"""

import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO

DATA_YAML = PROJECT_ROOT / "datasets" / "unified_plate_polygon" / "data.yaml"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
SAVE_PATH = WEIGHTS_DIR / "plate_polygon_detector.pt"
BACKUP_PATH = WEIGHTS_DIR / "plate_polygon_detector_thai_only.pt"

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def train_unified_detector(epochs=15, batch=16, imgsz=640):
    print("=======================================================")
    print("--- Training Unified Thai + Lao License Plate Detector ---")
    print(f"Device: {DEVICE} | Epochs: {epochs} | Batch: {batch} | Imgsz: {imgsz}")
    print(f"Dataset YAML: {DATA_YAML}")
    print("=======================================================\n")

    if not DATA_YAML.exists():
        print(f"Data YAML not found at {DATA_YAML}. Building dataset now...")
        import src.build_unified_plate_dataset as builder
        builder.main()

    # Backup existing checkpoint
    if SAVE_PATH.exists() and not BACKUP_PATH.exists():
        shutil.copy2(SAVE_PATH, BACKUP_PATH)
        print(f"Backed up previous Thai-only detector to: {BACKUP_PATH}")

    # Load YOLO11n-seg pretrained weights
    base_weights = "yolo11n-seg.pt"
    print(f"Initializing YOLO segmentation from: {base_weights}")
    model = YOLO(base_weights)

    # Train
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=DEVICE,
        workers=0,
        project=str(PROJECT_ROOT / "runs" / "unified_plate_seg"),
        name="train_unified",
        exist_ok=True,
        save=True,
        plots=False,
    )

    # Copy best weights to target location
    best_weights = PROJECT_ROOT / "runs" / "unified_plate_seg" / "train_unified" / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, SAVE_PATH)
        print(f"\n[Success] Unified plate detector saved to: {SAVE_PATH}")
    else:
        print(f"\n[Warning] Could not find {best_weights}, saving last checkpoint.")
        last_weights = PROJECT_ROOT / "runs" / "unified_plate_seg" / "train_unified" / "weights" / "last.pt"
        if last_weights.exists():
            shutil.copy2(last_weights, SAVE_PATH)


if __name__ == "__main__":
    train_unified_detector()
