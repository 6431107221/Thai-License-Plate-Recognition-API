
import argparse
from ultralytics import YOLO
from src.config import cfg
from pathlib import Path

def train_detection(epochs=100):
    """
    Train Model 1: Plate Detection
    Config:
    - Resize: 640x640
    - Augs: Brightness +/- 25%, Blur 1px, Noise
    """
    print("\nStarting Model 1: Detection Training...")
    
    # Path to data.yaml
    data_path = cfg.PROJECT_ROOT / "yolo_datasets" / "detection" / "data.yaml"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    model = YOLO("yolo11n.pt") # Load pretrained Nano model

    # Hyperparameters mapping to user request
    # Brightness +/- 25% -> hsv_v = 0.25
    
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=640,
        device=0 if cfg.DEVICE.type == 'cuda' else 'cpu',
        project="yolo_train_runs",
        name="det_model",
        exist_ok=True,
        # --- Augmentations ---
        hsv_v=0.25,        # Brightness (+/- 25%)
        erasing=0.0,       # No erasing
        mosaic=1.0,        # Use Mosaic
    )
    print("Detection Training Completed.")

def train_segmentation(epochs=100):
    """
    Train Model 2: Component Segmentation
    Config:
    - Resize: 320x320
    - Augs: Rotation +/- 5 deg, Blur
    """
    print("\nStarting Model 2: Segmentation Training...")
    
    # Path to data.yaml
    data_path = cfg.PROJECT_ROOT / "yolo_datasets" / "segmentation" / "data.yaml"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    model = YOLO("yolo11n-seg.pt") # Load pretrained Nano Segmentation model

    # Hyperparameters mapping
    # Rotation +/- 5 deg -> degrees = 5
    
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=320,         # User specified 320x320
        device=0 if cfg.DEVICE.type == 'cuda' else 'cpu',
        project="yolo_train_runs",
        name="seg_model",
        exist_ok=True,
        # --- Augmentations ---
        degrees=5.0,       # Rotation +/- 5 deg
        hsv_v=0.0,         # Disable brightness aug (user didn't ask for it here)
        mosaic=1.0,
    )
    print("Segmentation Training Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["detect", "segment", "all"], help="Which model to train")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    if args.task in ["detect", "all"]:
        train_detection(args.epochs)
        
    if args.task in ["segment", "all"]:
        train_segmentation(args.epochs)
