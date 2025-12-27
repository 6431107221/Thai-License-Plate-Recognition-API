
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- Config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Input: Where Roboflow dataset is
YOLO_SEG_ROOT = PROJECT_ROOT / "yolo_datasets" / "segmentation"
# Output: Where cropped images will be saved
OUTPUT_ROOT = PROJECT_ROOT / "crops_all"

# YOLO Class Mapping (Must match your data.yaml)
CLS_PLATE = 0    
CLS_PROV = 1     

def clean_filename(fname):
    """Normalize filename."""
    stem = Path(fname).stem
    if ".rf." in stem:
        stem = stem.split(".rf.")[0]
    return stem.lower().strip()

def yolo_to_bbox(line, img_w, img_h):
    """Convert YOLO format line (box or polygon) to [x1, y1, x2, y2]"""
    parts = line.split()
    cls = int(parts[0])
    
    if len(parts) > 5:
        # Polygon Format
        coords = list(map(float, parts[1:]))
        xs = coords[::2]
        ys = coords[1::2]
        x_min = min(xs) * img_w
        x_max = max(xs) * img_w
        y_min = min(ys) * img_h
        y_max = max(ys) * img_h
        return cls, [int(x_min), int(y_min), int(x_max), int(y_max)]
    else:
        # Detection Format
        x_c, y_c, w, h = map(float, parts[1:5])
        x1 = int((x_c - w/2) * img_w)
        y1 = int((y_c - h/2) * img_h)
        x2 = int((x_c + w/2) * img_w)
        y2 = int((y_c + h/2) * img_h)
        return cls, [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]

def process_split(split_name):
    img_dir = YOLO_SEG_ROOT / split_name / "images"
    lbl_dir = YOLO_SEG_ROOT / split_name / "labels"
    
    out_plate_dir = OUTPUT_ROOT / split_name / "plates"
    out_prov_dir = OUTPUT_ROOT / split_name / "provs"
    out_plate_dir.mkdir(parents=True, exist_ok=True)
    out_prov_dir.mkdir(parents=True, exist_ok=True)

    records = []
    
    # Gather Images
    img_files = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        img_files.extend(list(img_dir.glob(ext)))
    
    print(f"\nProcessing '{split_name}': Found {len(img_files)} images")
    
    count_processed = 0
    
    for img_path in tqdm(img_files):
        # 1. Image Path -> Label Path
        # Assumes label has same stem as image
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
        
        # 2. Extract Crops
        for line in lines:
            cls, bbox = yolo_to_bbox(line, w, h)
            # Safe crop with boundary check
            x1, y1, x2, y2 = bbox
            crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            
            if crop.size == 0: continue

            if cls == CLS_PLATE: 
                plate_crop = crop
            elif cls == CLS_PROV: 
                prov_crop = crop
        
        # 3. Save Crops and Record
        base_id = clean_filename(img_path.name)
        
        if plate_crop is not None:
            save_name_plate = f"{base_id}_plate.jpg"
            cv2.imwrite(str(out_plate_dir / save_name_plate), plate_crop)
            
            # Default empty labels (User will fill or use pre-existing logic later)
            record = {
                "image": f"{split_name}/plates/{save_name_plate}",
                "gt_plate": "",    # Placeholder
                "gt_province": ""  # Placeholder
            }
            
            if prov_crop is not None:
                save_name_prov = f"{base_id}_prov.jpg"
                cv2.imwrite(str(out_prov_dir / save_name_prov), prov_crop)
                
            records.append(record)
            count_processed += 1
            
    print(f"   Processed: {count_processed} / {len(img_files)}")
    return records

def main():
    print("Starting Data Preparation (Cropping Only)...")

    for split in ["train", "valid", "test"]:
        if (YOLO_SEG_ROOT / split).exists():
            records = process_split(split)
            
            if records:
                # Save CSV Template
                df = pd.DataFrame(records)
                csv_filename = f"{split}_unified.csv"
                if split == "valid": csv_filename = "val_unified.csv"
                
                out_csv_path = OUTPUT_ROOT / split / csv_filename
                df.to_csv(out_csv_path, index=False, encoding='utf-8-sig')
                print(f"   Saved CSV Template: {out_csv_path} ({len(df)} rows)")

    print("\nAll Done! Images cropped. Please fill in the CSV labels manually or via script.")

if __name__ == "__main__":
    main()
