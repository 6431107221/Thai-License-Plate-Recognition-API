"""
src/prepare_lao_annotation_set.py

Curates a balanced set of 400 Lao license plate images across all 18 provinces and colors,
generates draft pseudo-bounding box annotations (in standard YOLO format) with inverted
layout (Top: province, Bottom: plate_char), and writes a data.yaml ready for Roboflow,
LabelImg, or direct fine-tuning.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAO_DATASET_DIR = PROJECT_ROOT / "datasets" / "lao-plate-dataset"
SRC_IMAGES_DIR = LAO_DATASET_DIR / "images"
GT_CSV = LAO_DATASET_DIR / "ground_truth_all.csv"

OUT_DIR = PROJECT_ROOT / "datasets" / "lao_component_annotation"
OUT_IMAGES_DIR = OUT_DIR / "images"
OUT_LABELS_DIR = OUT_DIR / "labels"
OUT_VIS_DIR = OUT_DIR / "visualizations"

OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
OUT_VIS_DIR.mkdir(parents=True, exist_ok=True)

def create_lao_annotation_package(target_count: int = 400):
    df = pd.read_csv(GT_CSV)
    print(f"Loaded ground truth with {len(df)} records across {df['province'].nunique()} provinces.")

    # Stratified sampling across provinces
    sampled_dfs = []
    for prov, group in df.groupby("province"):
        # For thin classes take all (or up to 25); for large classes cap at 40
        n_samples = min(len(group), 35 if prov == "ນະຄອນຫຼວງວຽງຈັນ" else max(10, min(25, len(group))))
        sampled_dfs.append(group.sample(n=n_samples, random_state=42))

    sample_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # If still below target_count, sample extra from high-review items
    if len(sample_df) < target_count:
        rem = df[~df["filename"].isin(sample_df["filename"])]
        needed = target_count - len(sample_df)
        sample_df = pd.concat([sample_df, rem.sample(n=min(needed, len(rem)), random_state=42)], ignore_index=True)

    print(f"Sampled {len(sample_df)} representative Lao plates for annotation.")

    # YOLO Classes
    # 0: plate_char
    # 1: province
    vis_count = 0

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Generating YOLO annotation files"):
        fname = row["filename"]
        src_path = SRC_IMAGES_DIR / fname

        if not src_path.exists() or src_path.stat().st_size < 500:
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Copy image to annotation set
        dest_img_path = OUT_IMAGES_DIR / fname
        cv2.imwrite(str(dest_img_path), img)

        # Generate intelligent draft bounding boxes for Lao inverted layout:
        # Province is top: y ~ 0.08 to 0.36
        # Characters are bottom: y ~ 0.38 to 0.94
        prov_x1, prov_y1, prov_x2, prov_y2 = int(w * 0.12), int(h * 0.06), int(w * 0.88), int(h * 0.38)
        char_x1, char_y1, char_x2, char_y2 = int(w * 0.06), int(h * 0.38), int(w * 0.94), int(h * 0.94)

        # Convert to YOLO format: class_id x_center y_center width height (normalized)
        def to_yolo(x1, y1, x2, y2, img_w, img_h):
            xc = ((x1 + x2) / 2.0) / img_w
            yc = ((y1 + y2) / 2.0) / img_h
            bw = (x2 - x1) / float(img_w)
            bh = (y2 - y1) / float(img_h)
            return xc, yc, bw, bh

        c_xc, c_yc, c_w, c_h = to_yolo(char_x1, char_y1, char_x2, char_y2, w, h)
        p_xc, p_yc, p_w, p_h = to_yolo(prov_x1, prov_y1, prov_x2, prov_y2, w, h)

        base_stem = Path(fname).stem
        label_file = OUT_LABELS_DIR / f"{base_stem}.txt"

        with open(label_file, "w", encoding="utf-8") as f:
            # Class 0: plate_char
            f.write(f"0 {c_xc:.6f} {c_yc:.6f} {c_w:.6f} {c_h:.6f}\n")
            # Class 1: province
            f.write(f"1 {p_xc:.6f} {p_yc:.6f} {p_w:.6f} {p_h:.6f}\n")

        # Create visualization for first 16 samples
        if vis_count < 16:
            vis_img = img.copy()
            # Province: Cyan
            cv2.rectangle(vis_img, (prov_x1, prov_y1), (prov_x2, prov_y2), (255, 240, 0), 2)
            cv2.putText(vis_img, "province", (prov_x1, max(12, prov_y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 240, 0), 1)
            # Plate char: Amber
            cv2.rectangle(vis_img, (char_x1, char_y1), (char_x2, char_y2), (0, 165, 255), 2)
            cv2.putText(vis_img, "plate_char", (char_x1, max(12, char_y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            
            cv2.imwrite(str(OUT_VIS_DIR / f"vis_{fname}"), vis_img)
            vis_count += 1

    # Write data.yaml
    yaml_content = f"""# YOLOv11 Component Dataset for Lao License Plates
path: {OUT_DIR.absolute()}
train: images
val: images

names:
  0: plate_char
  1: province
"""
    with open(OUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # Save sampled metadata CSV
    sample_df.to_csv(OUT_DIR / "annotations_metadata.csv", index=False, encoding="utf-8")
    print(f"\nAnnotation package created at: {OUT_DIR}")
    print(f"  - Images: {len(list(OUT_IMAGES_DIR.glob('*.jpg')))}")
    print(f"  - Labels: {len(list(OUT_LABELS_DIR.glob('*.txt')))}")
    print(f"  - Config: {OUT_DIR / 'data.yaml'}")
    print(f"  - Visual verification samples: {OUT_VIS_DIR}")

if __name__ == "__main__":
    create_lao_annotation_package()
