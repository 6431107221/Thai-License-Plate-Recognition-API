"""
src/prepare_ground_truth_crops.py

Processes two external datasets using Model 1 (Plate Polygon Detector + Rectification)
and Model 2 (Character and Province Component Detector).
Samples every 3rd sorted image, crops plate characters and provinces into separate folders,
and outputs CSV files formatted for ground truth annotation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO
from src.config import cfg
from src.prepare_perspective_dataset import (
    extract_quad_corners,
    warp_perspective_plate,
    fine_deskew_plate,
)


def get_default_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_and_sort_images(dir_path: Path) -> list[Path]:
    """Recursively finds and sorts all valid image files in the given directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    # Sort deterministically by filename
    return sorted(images, key=lambda p: (p.name.lower(), str(p)))


def process_ground_truth_dataset(
    source1_dir: Path,
    source2_dir: Path,
    output_dir: Path,
    step: int = 3,
    m1_path: Path | None = None,
    m2_path: Path | None = None,
    conf_m1: float = 0.35,
    conf_m2: float = 0.25,
) -> pd.DataFrame:
    device = get_default_device()
    print(f"Using compute device: {device}")

    m1_path = Path(m1_path) if m1_path else (cfg.WEIGHTS_DIR / "plate_polygon_detector.pt")
    m2_path = Path(m2_path) if m2_path else (cfg.WEIGHTS_DIR / "component_detector.pt")

    if not m1_path.exists():
        raise FileNotFoundError(f"Model 1 weights not found at: {m1_path}")
    if not m2_path.exists():
        raise FileNotFoundError(f"Model 2 weights not found at: {m2_path}")

    print(f"Loading Model 1 from: {m1_path}")
    model1 = YOLO(str(m1_path))
    print(f"Loading Model 2 from: {m2_path}")
    model2 = YOLO(str(m2_path))

    # 1. Collect and sample images from both sources
    print(f"\nScanning Source 1: {source1_dir}")
    s1_all = find_and_sort_images(source1_dir)
    s1_sampled = s1_all[::step]
    print(f"  Source 1 total images: {len(s1_all)} -> Sampled (every {step}rd): {len(s1_sampled)}")

    print(f"\nScanning Source 2: {source2_dir}")
    s2_all = find_and_sort_images(source2_dir)
    s2_sampled = s2_all[::step]
    print(f"  Source 2 total images: {len(s2_all)} -> Sampled (every {step}rd): {len(s2_sampled)}")

    # Combine sampled items with source tag
    combined_items: list[tuple[Path, str]] = []
    for p in s1_sampled:
        combined_items.append((p, "plate-location02"))
    for p in s2_sampled:
        combined_items.append((p, "thai-car-plate-v2"))

    total_combined = len(combined_items)
    print(f"\nTotal combined images to process: {total_combined}")

    # Prepare output directories
    output_dir = Path(output_dir)
    dir_plates = output_dir / "plate_char"
    dir_provs = output_dir / "province"
    dir_rectified = output_dir / "rectified_plates"

    dir_plates.mkdir(parents=True, exist_ok=True)
    dir_provs.mkdir(parents=True, exist_ok=True)
    dir_rectified.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    tag_id = 1
    m1_detected_count = 0
    m2_detected_both = 0
    m2_fallback_count = 0

    print("\nStarting pipeline processing (Model 1 -> Rectification -> Model 2)...")
    for img_path, source_tag in tqdm(combined_items, desc="Cropping Dataset"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h_orig, w_orig = img.shape[:2]

        # --- Stage 1: Model 1 Plate Polygon Detection ---
        try:
            res1 = model1(img, conf=conf_m1, verbose=False, device=device)[0]
        except Exception as e:
            continue

        if len(res1.boxes) == 0:
            continue

        m1_detected_count += 1

        # Select highest-confidence plate detection
        best_box_idx = int(np.argmax(res1.boxes.conf.cpu().numpy()))

        rectified_plate = None

        if res1.masks is not None and len(res1.masks) > best_box_idx:
            poly = res1.masks.xy[best_box_idx].astype(np.float32)
            if len(poly) >= 3:
                quad = extract_quad_corners(poly, img=img)
                if quad is not None:
                    warped = warp_perspective_plate(
                        img, quad, target_width=320, target_height=160, padding_frac=0.08
                    )
                    rectified_plate = fine_deskew_plate(warped)

        # Fallback to standard bbox crop if polygon extraction failed
        if rectified_plate is None:
            x1, y1, x2, y2 = res1.boxes.xyxy[best_box_idx].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)
            raw_box = img[y1:y2, x1:x2]
            if raw_box.size > 0:
                rectified_plate = cv2.resize(raw_box, (320, 160), interpolation=cv2.INTER_CUBIC)

        if rectified_plate is None or rectified_plate.size == 0:
            continue

        rh, rw = rectified_plate.shape[:2]

        # --- Stage 2: Model 2 Component Detection ---
        char_crop = None
        prov_crop = None

        try:
            res2 = model2(rectified_plate, conf=conf_m2, verbose=False, device=device)[0]
            for box in res2.boxes:
                c_id = int(box.cls[0].item())
                c_name = model2.names[c_id].lower()
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)

                # Add a subtle 3px margin to avoid clipping character edges
                pad_x = 3
                pad_y = 2
                bx1, by1 = max(0, bx1 - pad_x), max(0, by1 - pad_y)
                bx2, by2 = min(rw, bx2 + pad_x), min(rh, by2 + pad_y)

                comp_crop = rectified_plate[by1:by2, bx1:bx2]
                if comp_crop.size == 0:
                    continue

                if ("plate" in c_name or "char" in c_name) and char_crop is None:
                    char_crop = comp_crop
                elif "prov" in c_name and prov_crop is None:
                    prov_crop = comp_crop
        except Exception as e:
            pass

        # Proportional fallbacks if Model 2 missed either component
        used_fallback = False
        if char_crop is None or char_crop.size == 0:
            char_crop = rectified_plate[0 : int(rh * 0.65), 0:rw]
            used_fallback = True
        if prov_crop is None or prov_crop.size == 0:
            prov_crop = rectified_plate[int(rh * 0.60) : rh, 0:rw]
            used_fallback = True

        if used_fallback:
            m2_fallback_count += 1
        else:
            m2_detected_both += 1

        # File naming convention
        base_name = f"{tag_id:06d}_{img_path.stem}"
        name_plate = f"{base_name}_plate.jpg"
        name_prov = f"{base_name}_prov.jpg"
        name_rect = f"{base_name}_rectified.jpg"

        # Save cropped images
        path_plate = dir_plates / name_plate
        path_prov = dir_provs / name_prov
        path_rect = dir_rectified / name_rect

        cv2.imwrite(str(path_plate), char_crop)
        cv2.imwrite(str(path_prov), prov_crop)
        cv2.imwrite(str(path_rect), rectified_plate)

        # Rel paths for CSV
        rel_plate = f"plate_char/{name_plate}"
        rel_prov = f"province/{name_prov}"
        rel_rect = f"rectified_plates/{name_rect}"

        records.append({
            "tag_id": tag_id,
            "plate_image": rel_plate,
            "province_image": rel_prov,
            "rectified_image": rel_rect,
            "original_image": str(img_path.relative_to(PROJECT_ROOT)) if img_path.is_relative_to(PROJECT_ROOT) else str(img_path),
            "source_dataset": source_tag,
            "gt_plate": "",
            "gt_province": "",
        })
        tag_id += 1

    df_unified = pd.DataFrame(records)

    # Save Unified CSV
    unified_csv_path = output_dir / "ground_truth_unified.csv"
    df_unified.to_csv(unified_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Saved] Unified Ground Truth CSV: {unified_csv_path} ({len(df_unified)} rows)")

    # Save Plate Characters CSV
    df_plate = df_unified[["tag_id", "plate_image", "original_image", "gt_plate"]].rename(
        columns={"plate_image": "image"}
    )
    plate_csv_path = output_dir / "gt_plate_char.csv"
    df_plate.to_csv(plate_csv_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] Plate Character Annotation CSV: {plate_csv_path} ({len(df_plate)} rows)")

    # Save Province CSV
    df_prov = df_unified[["tag_id", "province_image", "original_image", "gt_province"]].rename(
        columns={"province_image": "image"}
    )
    prov_csv_path = output_dir / "gt_province.csv"
    df_prov.to_csv(prov_csv_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] Province Annotation CSV: {prov_csv_path} ({len(df_prov)} rows)")

    print("\n=== Dataset Processing Summary ===")
    print(f"Total sampled images scanned : {total_combined}")
    print(f"Plates detected by Model 1   : {m1_detected_count} ({m1_detected_count / max(total_combined, 1) * 100:.1f}%)")
    print(f"Model 2 detected both boxes  : {m2_detected_both} ({m2_detected_both / max(m1_detected_count, 1) * 100:.1f}%)")
    print(f"Used proportional fallback   : {m2_fallback_count}")
    print(f"Total crop pairs produced    : {len(df_unified)}")

    return df_unified


def main():
    parser = argparse.ArgumentParser(description="Crop Thai LPR Ground Truth Dataset using Model 1 and Model 2")
    parser.add_argument(
        "--source1",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "Dataset_OTHER" / "dataset" / "plate-location02" / "images"),
        help="Path to Source 1 dataset directory",
    )
    parser.add_argument(
        "--source2",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "Dataset_OTHER" / "Thai Car Plate.v2-roboflow-instant-1--eval-.yolov8"),
        help="Path to Source 2 dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "output" / "ground_truth_crops"),
        help="Output directory for cropped images and CSV files",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=3,
        help="Sample every Nth sorted image (default: 3)",
    )
    args = parser.parse_args()

    process_ground_truth_dataset(
        source1_dir=Path(args.source1),
        source2_dir=Path(args.source2),
        output_dir=Path(args.output_dir),
        step=args.step,
    )


if __name__ == "__main__":
    main()
