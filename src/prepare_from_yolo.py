
import argparse
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

from src.prepare_perspective_dataset import run_pipeline

# --- Config & Defaults ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YOLO_DIR = PROJECT_ROOT / "datasets" / "LPR 2 - Polygon.yolov11"
LEGACY_YOLO_DIR = PROJECT_ROOT / "yolo_datasets" / "segmentation"
OUTPUT_ROOT = PROJECT_ROOT / "crops_all"

CLASS_PLATE = 0    
CLASS_PROV = 1     

def yolo_to_bbox(line, img_w, img_h):
    parts = line.split()
    class_id = int(parts[0])
    
    if len(parts) > 5:
        # Polygon Format
        coords = [float(x) for x in parts[1:]]
        xs = coords[::2]
        ys = coords[1::2]
        x_min = min(xs) * img_w
        x_max = max(xs) * img_w
        y_min = min(ys) * img_h
        y_max = max(ys) * img_h
        return class_id, [int(x_min), int(y_min), int(x_max), int(y_max)]
    else:
        # Square Format
        x_c, y_c, w, h = [float(x) for x in parts[1:5]]
        x1 = int((x_c - w/2) * img_w)
        y1 = int((y_c - h/2) * img_h)
        x2 = int((x_c + w/2) * img_w)
        y2 = int((y_c + h/2) * img_h)
        return class_id, [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]

def process_split_legacy(split_name, yolo_root=LEGACY_YOLO_DIR, output_root=OUTPUT_ROOT):
    img_dir = yolo_root / split_name / "images"
    lbl_dir = yolo_root / split_name / "labels"
    
    out_plate_dir = output_root / split_name / "plates"
    out_prov_dir = output_root / split_name / "provs"
    out_plate_dir.mkdir(parents=True, exist_ok=True)
    out_prov_dir.mkdir(parents=True, exist_ok=True)

    records = []
    
    img_files = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        img_files.extend(list(img_dir.glob(ext)))
    
    print(f"\nProcessing '{split_name}': Found {len(img_files)} images")
    
    count_processed = 0
    
    for img_path in tqdm(img_files):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        
        if not lbl_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w, _ = img.shape
        
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            
        plate_crop = None
        prov_crop = None
        
        for line in lines:
            cls, bbox = yolo_to_bbox(line, w, h)
            x1, y1, x2, y2 = bbox
            crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            
            if crop.size == 0: continue

            if cls == CLASS_PLATE: 
                plate_crop = crop
            elif cls == CLASS_PROV: 
                prov_crop = crop
        
        if plate_crop is not None:
            base_name = img_path.stem
            save_name_plate = f"{base_name}_plate.jpg"
            cv2.imwrite(str(out_plate_dir / save_name_plate), plate_crop)
            
            record = {
                "image": f"{split_name}/plates/{save_name_plate}",
                "gt_plate": "",    
                "gt_province": ""  
            }
            
            if prov_crop is not None:
                save_name_prov = f"{base_name}_prov.jpg"
                cv2.imwrite(str(out_prov_dir / save_name_prov), prov_crop)
                
            records.append(record)
            count_processed += 1
            
    print(f"   Processed: {count_processed} / {len(img_files)}")
    return records

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from YOLO segmentation / polygon annotations")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_YOLO_DIR if DEFAULT_YOLO_DIR.exists() else LEGACY_YOLO_DIR),
        help="Path to YOLO polygon dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_ROOT),
        help="Path to destination crops root directory",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "rectify", "components"],
        default="full",
        help="Processing mode: full, rectify (front view only), or components",
    )
    parser.add_argument(
        "--legacy-bbox",
        action="store_true",
        help="Use legacy axis-aligned square bounding box crop instead of perspective transform",
    )
    args = parser.parse_args()

    if args.legacy_bbox:
        dataset_path = Path(args.dataset_dir)
        output_path = Path(args.output_dir)
        for split in ["train", "valid", "test"]:
            if (dataset_path / split).exists():
                records = process_split_legacy(split, yolo_root=dataset_path, output_root=output_path)
                if records:
                    df = pd.DataFrame(records)
                    out_csv_path = output_path / split / f"{split}_unified.csv"
                    df.to_csv(out_csv_path, index=False, encoding='utf-8-sig')
                    print(f"   Saved Legacy CSV: {out_csv_path} ({len(df)} rows)")
    else:
        run_pipeline(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            mode=args.mode,
        )

if __name__ == "__main__":
    main()

