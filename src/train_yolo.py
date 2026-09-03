import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO
from src.config import cfg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEG_DATA = PROJECT_ROOT / "datasets" / "LPR 2 - Polygon.yolov11" / "data.yaml"


def get_default_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_detection(epochs=100, data=None, imgsz=640, batch=16, model_name="yolo11n.pt"):
    print("\n=== Training Model 1: License Plate Detection ===")
    data_path = Path(data) if data else (cfg.PROJECT_ROOT / "yolo_datasets" / "detection" / "data.yaml")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    device = get_default_device()
    print(f"Device: {device} | Dataset: {data_path} | Epochs: {epochs}")

    model = YOLO(model_name)
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="yolo_train_runs",
        name="det_model",
        exist_ok=True,
    )
    return results


def train_segmentation(
    epochs=50,
    data=None,
    imgsz=640,
    batch=16,
    model_name="yolo11n-seg.pt",
    save_name="plate_polygon_detector.pt",
):
    print("\n=== Training YOLO11 Polygon/Segmentation for Tilted Plates ===")
    data_path = Path(data) if data else DEFAULT_SEG_DATA
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    device = get_default_device()
    print(f"Device: {device} | Dataset: {data_path} | Epochs: {epochs}")

    model = YOLO(model_name)
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="yolo_train_runs",
        name="plate_polygon_seg",
        exist_ok=True,
    )

    # Copy best weights to weights/ directory
    candidates = []
    if hasattr(results, "save_dir") and results.save_dir:
        candidates.append(Path(results.save_dir) / "weights" / "best.pt")
    candidates.extend([
        Path("runs/segment/yolo_train_runs/plate_polygon_seg/weights/best.pt"),
        Path("yolo_train_runs/plate_polygon_seg/weights/best.pt"),
    ])

    best_weights = None
    for cand in candidates:
        if cand.exists():
            best_weights = cand
            break

    if best_weights:
        cfg.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        target_path = cfg.WEIGHTS_DIR / save_name
        import shutil
        shutil.copy(best_weights, target_path)
        print(f"\nSaved Best Polygon Model Weights to: {target_path}")
        if save_name != "plate_polygon_detector.pt":
            default_path = cfg.WEIGHTS_DIR / "plate_polygon_detector.pt"
            shutil.copy(best_weights, default_path)
            print(f"Updated default weights at: {default_path}")
    else:
        print("\n[Warning] Could not find best.pt to copy to weights directory.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO Detection or Polygon Segmentation for Thai LPR")
    parser.add_argument(
        "--task",
        type=str,
        default="segment",
        choices=["detect", "segment", "all"],
        help="Which task to train (default: segment)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_SEG_DATA),
        help="Path to dataset data.yaml (default: datasets/LPR 2 - Polygon.yolov11/data.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s-seg.pt",
        help="Base model pretrained weights (e.g. yolo11n-seg.pt or yolo11s-seg.pt)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument(
        "--save-name",
        type=str,
        default="plate_polygon_detector.pt",
        help="Target filename inside weights/ directory (default: plate_polygon_detector.pt)",
    )
    args = parser.parse_args()

    if args.task in ["detect", "all"]:
        train_detection(epochs=args.epochs, data=args.data, imgsz=args.imgsz, batch=args.batch, model_name=args.model)

    if args.task in ["segment", "all"]:
        train_segmentation(
            epochs=args.epochs,
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            model_name=args.model,
            save_name=args.save_name,
        )