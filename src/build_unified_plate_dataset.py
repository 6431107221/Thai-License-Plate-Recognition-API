"""
src/build_unified_plate_dataset.py

Merges Thai and Lao vehicle datasets into a single unified YOLO11 segmentation dataset:
1. Thai Polygon: datasets/Thai/LPR 2 - Polygon.yolov11_new (1,068 train, 305 val)
2. Lao Plate v3: datasets/Lao/laos plate.v3i.yolov11 (660 train, 15 val)
3. Lao Plate v2: datasets/Lao/Lao License Plates.v2i.yolov11 (1,089 train, 104 val)

Converts Lao bounding boxes (cls cx cy w h) to 4-corner polygon coordinates (cls x1 y1 x2 y2 x3 y3 x4 y4)
so YOLO11-seg learns pixel-level polygonal perimeter boundaries on both Thai and Lao plates.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_DIR = PROJECT_ROOT / "datasets" / "unified_plate_polygon"

THAI_DIR = PROJECT_ROOT / "datasets" / "Thai" / "LPR 2 - Polygon.yolov11_new"
LAO_V3_DIR = PROJECT_ROOT / "datasets" / "Lao" / "laos plate.v3i.yolov11"
LAO_V2_DIR = PROJECT_ROOT / "datasets" / "Lao" / "Lao License Plates.v2i.yolov11"


def convert_bbox_to_polygon_line(line: str) -> str:
    """Converts 'cls cx cy w h' -> 'cls x1 y1 x2 y2 x3 y3 x4 y4' if needed."""
    parts = line.strip().split()
    if len(parts) == 5:
        cls_id = parts[0]
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1, y1 = max(0.0, cx - w/2), max(0.0, cy - h/2)
        x2, y2 = min(1.0, cx + w/2), max(0.0, cy - h/2)
        x3, y3 = min(1.0, cx + w/2), min(1.0, cy + h/2)
        x4, y4 = max(0.0, cx - w/2), min(1.0, cy + h/2)
        return f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
    elif len(parts) >= 9:
        # Already polygon format, just ensure class is 0
        return f"0 {' '.join(parts[1:])}"
    return ""


def main():
    print(f"=== Building Unified Plate Polygon Dataset at {UNIFIED_DIR} ===")
    if UNIFIED_DIR.exists():
        shutil.rmtree(UNIFIED_DIR)

    for split in ["train", "valid"]:
        (UNIFIED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (UNIFIED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "valid": 0}

    # 1. Ingest Thai Polygon
    for split in ["train", "valid"]:
        t_img_dir = THAI_DIR / split / "images"
        t_lbl_dir = THAI_DIR / split / "labels"
        if not t_img_dir.exists(): continue
        for img_p in t_img_dir.glob("*.jpg"):
            lbl_p = t_lbl_dir / (img_p.stem + ".txt")
            if not lbl_p.exists(): continue

            dest_img = UNIFIED_DIR / split / "images" / f"thai_{img_p.name}"
            dest_lbl = UNIFIED_DIR / split / "labels" / f"thai_{lbl_p.name}"
            shutil.copy2(img_p, dest_img)

            # Re-write label with class 0
            with open(lbl_p, "r") as f_in, open(dest_lbl, "w") as f_out:
                for line in f_in:
                    poly_line = convert_bbox_to_polygon_line(line)
                    if poly_line:
                        f_out.write(poly_line + "\n")
            counts[split] += 1

    # 2. Ingest Lao datasets
    for ds_name, ds_dir in [("laov3", LAO_V3_DIR), ("laov2", LAO_V2_DIR)]:
        for split in ["train", "valid"]:
            src_split = split
            s_img_dir = ds_dir / src_split / "images"
            s_lbl_dir = ds_dir / src_split / "labels"
            if not s_img_dir.exists(): continue

            for img_p in s_img_dir.glob("*.jpg"):
                lbl_p = s_lbl_dir / (img_p.stem + ".txt")
                if not lbl_p.exists(): continue

                dest_img = UNIFIED_DIR / split / "images" / f"{ds_name}_{img_p.name}"
                dest_lbl = UNIFIED_DIR / split / "labels" / f"{ds_name}_{lbl_p.name}"
                shutil.copy2(img_p, dest_img)

                with open(lbl_p, "r") as f_in, open(dest_lbl, "w") as f_out:
                    for line in f_in:
                        poly_line = convert_bbox_to_polygon_line(line)
                        if poly_line:
                            f_out.write(poly_line + "\n")
                counts[split] += 1

    # 3. Write data.yaml
    data_yaml_content = f"""# Unified Thai + Lao License Plate Polygon Segmentation Dataset
path: {UNIFIED_DIR.resolve()}
train: train/images
val: valid/images

nc: 1
names: ['plate']
"""
    with open(UNIFIED_DIR / "data.yaml", "w") as f:
        f.write(data_yaml_content)

    print(f"Dataset build complete! Train: {counts['train']} images, Valid: {counts['valid']} images.")
    print(f"data.yaml written to: {UNIFIED_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
