"""
src/verify_model2.py
Verification script for Model 2 (Character and Province Component Detector).
Runs inference on test rectified plates and visualizes detected bounding boxes.
"""

import sys
from pathlib import Path
import glob
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from src.config import cfg


def verify_component_detector(
    weights_path=None,
    test_dir=None,
    output_dir=None,
    num_samples=8,
    conf_thresh=0.25,
):
    weights_path = Path(weights_path) if weights_path else (cfg.WEIGHTS_DIR / "component_detector.pt")
    if not weights_path.exists():
        print(f"[Error] Model weights not found at: {weights_path}")
        return []

    print(f"Loading Model 2 from: {weights_path}")
    model = YOLO(str(weights_path))

    if test_dir is None:
        # Check Roboflow test set first, then valid set, then rectified plates output
        candidates = [
            PROJECT_ROOT / "datasets" / "LPR 2 - Charactor Detection.yolov11" / "test" / "images",
            PROJECT_ROOT / "datasets" / "LPR 2 - Charactor Detection.yolov11" / "valid" / "images",
            PROJECT_ROOT / "output" / "train" / "rectified_plates",
        ]
        for c in candidates:
            if c.exists() and len(list(c.glob("*.jpg")) + list(c.glob("*.png"))) > 0:
                test_dir = c
                break

    if not test_dir or not Path(test_dir).exists():
        print("[Error] No test image directory found.")
        return []

    print(f"Testing on images from: {test_dir}")
    image_paths = sorted(list(Path(test_dir).glob("*.jpg")) + list(Path(test_dir).glob("*.png")))
    if not image_paths:
        print("[Error] No images found in test directory.")
        return []

    # Select representative samples
    selected_paths = image_paths[:num_samples]
    output_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "output" / "model2_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_images = []
    colors = {
        "plate_char": (255, 140, 0),    # Bright orange/cyan (BGR: (0, 165, 255) in BGR)
        "province": (0, 220, 100),       # Vibrant Green
    }
    default_color = (200, 200, 0)

    for idx, img_p in enumerate(selected_paths):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue

        results = model.predict(img_bgr, conf=conf_thresh, imgsz=320, verbose=False)[0]
        vis = img_bgr.copy()
        h, w = vis.shape[:2]

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, f"cls_{cls_id}")
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            color = colors.get(cls_name, default_color)
            # Draw box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw label banner
            label = f"{cls_name} {conf:.2f}"
            font_scale = 0.45
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Put label inside or above
            label_y = max(y1 - 4, th + 2)
            cv2.rectangle(vis, (x1, label_y - th - 2), (x1 + tw + 2, label_y + baseline), color, -1)
            cv2.putText(vis, label, (x1 + 1, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        out_name = f"verify_{idx+1:02d}_{img_p.stem}.jpg"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), vis)
        saved_images.append(out_path)
        print(f"Saved: {out_path.name} ({len(results.boxes)} detections)")

    print(f"\nVerification complete! {len(saved_images)} images saved to: {output_dir}")
    return saved_images


if __name__ == "__main__":
    verify_component_detector()
