import argparse
from ultralytics import YOLO
from src.config import cfg
from pathlib import Path

def train_detection(epochs=100):
    print("\nStarting Model 1: Detection")
    
    data_path = cfg.PROJECT_ROOT / "yolo_datasets" / "detection" / "data.yaml"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=640,
        device=0 if cfg.DEVICE.type == 'cuda' else 'cpu',
        project="yolo_train_runs",
        name="det_model",
        exist_ok=True,
        hsv_v=0.25,
        erasing=0.0,
        mosaic=1.0,
    )
    print("Detection Training Completed.")

def train_segmentation(epochs=100):
    print("\nStarting Model 2: Segmentation")
    
    data_path = cfg.PROJECT_ROOT / "yolo_datasets" / "segmentation" / "data.yaml"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    model = YOLO("yolo11n-seg.pt")

    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=320,
        device=0 if cfg.DEVICE.type == 'cuda' else 'cpu',
        project="yolo_train_runs",
        name="seg_model",
        exist_ok=True,
        degrees=5.0,
        hsv_v=0.0,
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