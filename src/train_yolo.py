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
DEFAULT_COMPONENT_DATA = PROJECT_ROOT / "datasets" / "LPR 2 - Charactor Detection.yolov11" / "data.yaml"


def get_default_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_detection(
    epochs=100,
    data=None,
    imgsz=640,
    batch=16,
    model_name="yolo11n.pt",
    save_name="plate_detector.pt",
    name="det_model",
):
    print("\n=== Training YOLO Detection ===")
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
        name=name,
        exist_ok=True,
    )

    # Copy best weights to weights/ directory
    candidates = []
    if hasattr(results, "save_dir") and results.save_dir:
        candidates.append(Path(results.save_dir) / "weights" / "best.pt")
    candidates.extend([
        Path(f"runs/detect/yolo_train_runs/{name}/weights/best.pt"),
        Path(f"yolo_train_runs/{name}/weights/best.pt"),
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
        print(f"\nSaved Best Detection Model Weights to: {target_path}")
        if save_name == "component_detector.pt":
            alias_path = cfg.WEIGHTS_DIR / "plate_character_detector.pt"
            shutil.copy(best_weights, alias_path)
            print(f"Copied alias to: {alias_path}")
    else:
        print("\n[Warning] Could not find best.pt to copy to weights directory.")

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


def train_component_detection(
    epochs=60,
    data=None,
    imgsz=320,
    batch=16,
    model_name="yolo11s.pt",
    save_name="component_detector.pt",
):
    print("\n=== Training Model 2: License Plate Character & Province Detector ===")
    data_path = Path(data) if data else DEFAULT_COMPONENT_DATA
    return train_detection(
        epochs=epochs,
        data=data_path,
        imgsz=imgsz,
        batch=batch,
        model_name=model_name,
        save_name=save_name,
        name="component_detector",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO Detection, Component Detection, or Polygon Segmentation")
    parser.add_argument(
        "--task",
        type=str,
        default="component",
        choices=["detect", "segment", "component", "all"],
        help="Which task to train (default: component)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset data.yaml (defaults to task-specific dataset)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Base model pretrained weights (e.g. yolo11s.pt or yolo11s-seg.pt)",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size")
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Target filename inside weights/ directory",
    )
    args = parser.parse_args()

    if args.task == "component":
        data = args.data or str(DEFAULT_COMPONENT_DATA)
        model = args.model or "yolo11s.pt"
        imgsz = args.imgsz or 320
        save_name = args.save_name or "component_detector.pt"
        train_component_detection(
            epochs=args.epochs,
            data=data,
            imgsz=imgsz,
            batch=args.batch,
            model_name=model,
            save_name=save_name,
        )

    elif args.task == "detect":
        data = args.data or str(cfg.PROJECT_ROOT / "yolo_datasets" / "detection" / "data.yaml")
        model = args.model or "yolo11n.pt"
        imgsz = args.imgsz or 640
        save_name = args.save_name or "plate_detector.pt"
        train_detection(
            epochs=args.epochs,
            data=data,
            imgsz=imgsz,
            batch=args.batch,
            model_name=model,
            save_name=save_name,
        )

    elif args.task == "segment":
        data = args.data or str(DEFAULT_SEG_DATA)
        model = args.model or "yolo11s-seg.pt"
        imgsz = args.imgsz or 640
        save_name = args.save_name or "plate_polygon_detector.pt"
        train_segmentation(
            epochs=args.epochs,
            data=data,
            imgsz=imgsz,
            batch=args.batch,
            model_name=model,
            save_name=save_name,
        )

    elif args.task == "all":
        # Train both Model 1 (segment) and Model 2 (component)
        train_segmentation(
            epochs=args.epochs,
            data=str(DEFAULT_SEG_DATA),
            imgsz=640,
            batch=args.batch,
            model_name="yolo11s-seg.pt",
            save_name="plate_polygon_detector.pt",
        )
        train_component_detection(
            epochs=args.epochs,
            data=str(DEFAULT_COMPONENT_DATA),
            imgsz=320,
            batch=args.batch,
            model_name="yolo11s.pt",
            save_name="component_detector.pt",
        )