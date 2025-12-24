import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- Config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_SEG_ROOT = PROJECT_ROOT / "yolo_datasets" / "segmentation"
OUTPUT_ROOT = PROJECT_ROOT / "crops_all"
MASTER_LABELS_PATH = PROJECT_ROOT / "traintestvalid.csv"

# YOLO Class Mapping (Must match data.yaml)
CLS_PLATE = 0    
CLS_PROV = 1     

def clean_filename(fname):
    """Normalize filename to a common ID for matching."""
    stem = Path(fname).stem
    
    if ".rf." in stem:
        stem = stem.split(".rf.")[0]
        
    lower_stem = stem.lower()
    if lower_stem.endswith("_jpg"):
        stem = stem[:-4]
    elif lower_stem.endswith("_png"):
        stem = stem[:-4]
    elif lower_stem.endswith("_jpeg"):
        stem = stem[:-5]
        
    return stem.strip()

def load_master_labels():
    """Load master CSV labels into a dictionary."""
    if not MASTER_LABELS_PATH.exists():
        print(f"Error: Master Label file not found at {MASTER_LABELS_PATH}")
        return None
    
    try:
        df = pd.read_csv(MASTER_LABELS_PATH)
        print(f"Loaded Master CSV: {len(df)} rows.")

        label_map = {}
        for index, row in df.iterrows():
            raw_fname = str(row.iloc[0]).strip()
            base_id = clean_filename(raw_fname)
            
            plate_txt = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else "" 
            prov_txt = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
            
            label_map[base_id] = {
                "gt_plate": plate_txt,
                "gt_province": prov_txt,
                "original_csv_name": raw_fname
            }
        
        return label_map
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def yolo_to_bbox(line, img_w, img_h):
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

def process_split(split_name, label_map):
    img_dir = YOLO_SEG_ROOT / split_name / "images"
    lbl_dir = YOLO_SEG_ROOT / split_name / "labels"
    
    out_plate_dir = OUTPUT_ROOT / split_name / "plates"
    out_prov_dir = OUTPUT_ROOT / split_name / "provs"
    out_plate_dir.mkdir(parents=True, exist_ok=True)
    out_prov_dir.mkdir(parents=True, exist_ok=True)

    records = []
    
    img_files = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        img_files.extend(list(img_dir.glob(ext)))
    
    print(f"\nProcessing '{split_name}': Found {len(img_files)} images")
    
    count_matched = 0
    count_fail = 0
    
    for img_path in tqdm(img_files):
        lookup_key = clean_filename(img_path.name)
        
        lb_filename = f"{img_path.stem}.txt"
        lbl_path = lbl_dir / lb_filename
        
        if not lbl_path.exists(): continue
        
        gt_data = label_map.get(lookup_key)
        
        if not gt_data:
            if count_fail < 3:
                print(f"      Debug: Unmatched Image '{lookup_key}' (Orig: {img_path.name})")
            count_fail += 1
            continue
            
        count_matched += 1
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w, _ = img.shape
        
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            
        plate_crop = None
        prov_crop = None
        
        for line in lines:
            cls, bbox = yolo_to_bbox(line, w, h)
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            if crop.size == 0: continue

            if cls == CLS_PLATE: 
                plate_crop = crop
            elif cls == CLS_PROV: 
                prov_crop = crop
        
        if plate_crop is not None:
            save_name = f"{lookup_key}_plate.jpg"
            cv2.imwrite(str(out_plate_dir / save_name), plate_crop)
            
            if prov_crop is not None:
                prov_save_name = f"{lookup_key}_prov.jpg"
                cv2.imwrite(str(out_prov_dir / prov_save_name), prov_crop)
            
            records.append({
                "image": f"{split_name}/plates/{save_name}",
                "gt_plate": gt_data["gt_plate"],
                "gt_province": gt_data["gt_province"]
            })
            
    print(f"   Matched & Processed: {count_matched} / {len(img_files)}")
    return records

def main():
    print("Starting Data Preparation from YOLO GT...")
    label_map = load_master_labels()
    if not label_map:
        return

    for split in ["train", "valid", "test"]:
        if (YOLO_SEG_ROOT / split).exists():
            records = process_split(split, label_map)
            
            if records:
                df = pd.DataFrame(records)
                
                csv_filename = f"{split}_unified.csv"
                if split == "valid": csv_filename = "val_unified.csv"
                
                out_csv_path = OUTPUT_ROOT / split / csv_filename
                df.to_csv(out_csv_path, index=False, encoding='utf-8-sig')
                print(f"   Saved CSV: {out_csv_path} ({len(df)} rows)")

    print("\nAll Done! Ready for OCR Training.")

if __name__ == "__main__":
    main()
