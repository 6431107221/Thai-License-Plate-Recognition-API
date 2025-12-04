import os
from pathlib import Path
import json
from PIL import Image
import tqdm
import pandas as pd
import re

# --- CONFIG ---
SOURCE_ROOT = Path("OCR-Car-Plate_dataset") 
CROPS_DIR = Path("crops_all") 
img_extension = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def main():
    # 1. ตรวจสอบโฟลเดอร์ Source
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ {SOURCE_ROOT} กรุณาตรวจสอบว่าวางโฟลเดอร์ Dataset ไว้ถูกต้อง")
    
    # 2. Crop Images
    for tvt in ["train", "valid", "test"]:
        img_dir = SOURCE_ROOT / tvt / "Images"
        anno_dir = SOURCE_ROOT / tvt / "_annotations.coco.json"

        # สร้างโฟลเดอร์ปลายทาง
        plate_out = CROPS_DIR / tvt / "plates"
        province_out = CROPS_DIR / tvt / "provs"
        plate_out.mkdir(parents=True, exist_ok=True)
        province_out.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            print(f"Skipping {tvt} (No images found)")
            continue

        # โหลด Annotations
        id_anno = {}
        id_cate = {}
        id_img_map = {} # Map id -> filename

        if anno_dir.exists():
            data = json.load(open(anno_dir, 'r', encoding='utf-8'))
            id_cate = {c["id"]: c["name"].lower() for c in data.get("categories", [])}
            # สร้าง Map: ImageID -> Filename
            id_img_map = {im['id']: im['file_name'] for im in data.get('images', [])}
            # Group Annotations by ImageID
            for a in data.get("annotations", []):
                id_anno.setdefault(a["image_id"], []).append(a)

        # หาไฟล์รูปภาพทั้งหมด
        images = [p for p in img_dir.rglob("*") if p.suffix.lower() in img_extension]
        print(f"[{tvt}] Found {len(images)} images in source.")

        # --- เริ่ม Loop Crop ---
        count_skipped = 0
        count_cropped = 0
        
        for p in tqdm.tqdm(images, desc=f'Crop {tvt}'):
            # สร้างชื่อไฟล์ปลายทางล่วงหน้าเพื่อตรวจสอบว่ามีหรือยัง
            # Format: {split}__{original_stem}__plate{suffix}
            out_name_plate = f"{tvt}__{p.stem}__plate{p.suffix}"
            out_path_plate = plate_out / out_name_plate

            #  เช็ค: ถ้ามีไฟล์แล้ว ให้ข้าม (Skip)
            if out_path_plate.exists():
                count_skipped += 1
                continue

            try:
                # ถ้ายังไม่มี ค่อยโหลดภาพ
                img = Image.open(p).convert('RGB')
                W, H = img.size
                
                # หา Image ID จากชื่อไฟล์ (ต้องตรงกับใน JSON)
                # Logic: วนหาว่าชื่อไฟล์ p.name ตรงกับ file_name ใน json หรือไม่
                img_id = None
                for k_id, v_fname in id_img_map.items():
                    if v_fname == p.name:
                        img_id = k_id
                        break
                
                plate_bbox = None; province_bbox = None 

                if img_id and img_id in id_anno:
                    for a in id_anno[img_id]:
                        cname = id_cate.get(a["category_id"], "")
                        if "plate" in cname or "ทะเบียน" in cname or "license" in cname:
                            plate_bbox = a["bbox"]
                        if "prov" in cname or "จังหวัด" in cname or "province" in cname:
                            province_bbox = a["bbox"]

                # Save Crop Image (Plate)
                saved_plate = False
                if plate_bbox:
                    x, y, w, h = plate_bbox
                    # ขยายขอบเล็กน้อยหรือ Crop ตาม BBox
                    crop = img.crop((int(x), int(y), int(x+w), int(y+h))).convert("L")
                    if crop.width > 2 and crop.height > 2:
                        crop.save(out_path_plate)
                        saved_plate = True

                # Save Crop Image (Province)
                if province_bbox:
                    x, y, w, h = province_bbox
                    crop = img.crop((int(x), int(y), int(x+w), int(y+h))).convert("RGB")
                    if crop.width > 2 and crop.height > 2:
                        out_name_prov = f"{tvt}__{p.stem}__prov{p.suffix}"
                        crop.save(province_out / out_name_prov)

                # Fallback (ถ้าไม่มี Plate ใช้ตรงกลางล่าง)
                if not saved_plate:
                    out_name_fallback = f"{tvt}__{p.stem}__plate_fallback{p.suffix}"
                    # Fallback Logic
                    cw, ch = int(W*0.45), int(H*0.14)
                    x0 = (W-cw)//2; y0 = int(H*0.7)
                    crop = img.crop((x0, y0, x0+cw, y0+ch)).convert("L")
                    crop.save(plate_out / out_name_fallback)
                
                count_cropped += 1

            except Exception as e:
                print(f"Error processing {p}: {e}")
                continue
        
        print(f"   -> Cropped: {count_cropped}, Skipped (Already existed): {count_skipped}")

    # 3. Match CSV (Simplified Logic)
    print("\nGenerating CSVs inside crops_all folders...")
    
    # กำหนดคู่ (Input CSV, Output Filename, Split Folder Name)
    # Output Name เป็นแค่ชื่อไฟล์ เพราะเราจะเอาไปแปะใน folder crops_all/{split}/
    DATA_SETS = [
        (SOURCE_ROOT/"train"/"train_data.csv", "train_unified.csv", "train"),
        (SOURCE_ROOT/"valid"/"valid_data.csv", "val_unified.csv",   "valid"),
        (SOURCE_ROOT/"test"/"test_data.csv",   "test_unified.csv",  "test")
    ]

    for inp, out_name, split in DATA_SETS:
        # Save path ใหม่: crops_all/{split}/{out_name}
        out_path = CROPS_DIR / split / out_name
        process_mapping_simple(inp, out_path, split)

def process_mapping_simple(csv_path, out_path, split_folder):
    if not csv_path.exists():
        print(f" CSV Not found: {csv_path}")
        return

    print(f"\nProcessing: {csv_path.name} -> {out_path}")
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    # โฟลเดอร์ที่มีรูป Crop แล้ว
    target_dir = CROPS_DIR / split_folder / "plates"
    # หาไฟล์ทั้งหมดในนั้น
    plate_files = list(target_dir.glob("*")) # เอาทุกไฟล์ใน folder
    print(f"  Found {len(plate_files)} crops in {target_dir}")

    # สร้าง Lookup Table: Original Stem -> Crop Path
    # ตัวอย่างชื่อไฟล์ Crop: "train__image_01__plate.jpg"
    # เราจะแยกด้วย "__" เพื่อเอาตรงกลางคือ "image_01" มาเป็น Key
    stem_to_crop = {}
    for p in plate_files:
        parts = p.name.split("__")
        if len(parts) >= 3:
            # parts[0] = split (train), parts[1] = stem, parts[2] = suffix
            # แต่บางทีชื่อไฟล์เดิมอาจมี underscore เราต้องระวัง
            # วิธีที่ดีกว่า: ตัด prefix "{split}__" และ suffix "__plate{ext}" ออก
            prefix = f"{split_folder}__"
            if p.name.startswith(prefix):
                temp = p.name[len(prefix):] # ตัด prefix
                # หาตำแหน่งของ "__plate" สุดท้าย
                idx = temp.rfind("__plate")
                if idx != -1:
                    stem = temp[:idx] # ได้ stem แท้ๆ แล้ว
                    stem_to_crop[stem.lower()] = p

    # ค้นหา Column
    cols_map = {c.lower():c for c in df.columns}
    fname_col = next((cols_map[c] for c in cols_map if any(x in c for x in ["file","image","name"])), None)
    plate_col = next((cols_map[c] for c in cols_map if any(x in c for x in ["plate","label","gt"])), None)
    prov_col  = next((cols_map[c] for c in cols_map if any(x in c for x in ["prov","จังหวัด"])), None)

    if not fname_col:
        print("Cannot find filename column in CSV")
        return

    rows = []
    missed_count = 0

    for i, r in df.iterrows():
        orig_name = str(r[fname_col]).strip()
        gt_plate = str(r[plate_col]).strip() if plate_col else ""
        gt_prov  = str(r[prov_col]).strip() if prov_col else ""

        # แปลงชื่อใน CSV ให้เป็น Stem (ตัดนามสกุลออก)
        orig_stem = Path(orig_name).stem.lower()

        # Match!
        if orig_stem in stem_to_crop:
            crop_path = stem_to_crop[orig_stem]
            # Save Relative Path (เพื่อให้ DataLoader หาเจอ)
            # ต้อง relative กับ CROPS_DIR (เพราะใน Dataset class เราเอา root มาต่อ)
            # แต่เดี๋ยวก่อน! ปกติ Dataset class จะรับ root.
            # ถ้าเราเก็บ csv ไว้ใน crops_all/train/ เราควรใช้ path ที่ relative กับ crops_all หรือ absolute?
            # Standard: relative to CROPS_DIR (root)
            rel_path = crop_path.relative_to(CROPS_DIR)
            rows.append({
                "image": str(rel_path).replace("\\", "/"), # Fix windows path separator
                "gt_plate": gt_plate, 
                "gt_province": gt_prov
            })
        else:
            missed_count += 1

    # Save CSV
    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  Saved {len(rows)} rows to {out_path}")
    else:
        print("No matches found! Check filename logic.")
    
    if missed_count > 0:
        print(f"  (Missed {missed_count} files)")

if __name__ == "__main__":
    main()