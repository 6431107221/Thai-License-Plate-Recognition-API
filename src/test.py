
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
import editdistance
from tqdm.auto import tqdm
import torch.nn.functional as F

# Import Local Modules
from config import cfg
from models import ResNetCRNN, ProvinceClassifier, best_path_decode
from preprocess import get_ocr_transforms, get_prov_transforms

def main():
    print(f"=== Starting Evaluation on Test Set ===")
    print(f"Device: {cfg.DEVICE}")

    # 1. Load Models & Metadata from .pth files (The Triple Truth)
    
    # --- OCR ---
    print("Loading OCR Model and Embedded Char Map...")
    if not cfg.OCR_MODEL_SAVE_PATH.exists():
        print(f"Error: OCR model not found at {cfg.OCR_MODEL_SAVE_PATH}")
        return
        
    ocr_ckpt = torch.load(cfg.OCR_MODEL_SAVE_PATH, map_location=cfg.DEVICE)
    
    # ดึง Char Map จากห่อ (ถ้าไม่มีค่อยไปหาไฟล์แยก)
    if isinstance(ocr_ckpt, dict) and "int_to_char" in ocr_ckpt:
        int_to_char = ocr_ckpt["int_to_char"]
        print(f" ✅ Using EMBEDDED char map from checkpoint ({len(int_to_char)} chars)")
    else:
        print(" ⚠️ No embedded char map found, falling back to disk...")
        with open(cfg.CHAR_MAP_PATH, 'r', encoding='utf-8') as f:
            int_to_char = json.load(f)

    # ดึง Weights จากห่อ
    if isinstance(ocr_ckpt, dict):
        state_dict = ocr_ckpt.get("model_state_dict", ocr_ckpt.get("model", ocr_ckpt))
    else:
        state_dict = ocr_ckpt
        
    ocr_model = ResNetCRNN(img_channel=1, num_classes=len(int_to_char)).to(cfg.DEVICE)
    ocr_model.load_state_dict(state_dict)
    ocr_model.eval()
    print(" ✅ OCR Model Weights Loaded")

    # --- Province ---
    print("Loading Province Model and Embedded Class Map...")
    if not cfg.PROV_MODEL_SAVE_PATH.exists():
        print(f"Error: Province model not found at {cfg.PROV_MODEL_SAVE_PATH}")
        return
        
    prov_ckpt = torch.load(cfg.PROV_MODEL_SAVE_PATH, map_location=cfg.DEVICE)
    
    # ดึง Class Map จากห่อ
    if isinstance(prov_ckpt, dict) and "class_map" in prov_ckpt:
        class_map = prov_ckpt["class_map"]
        # Ensure correct mapping {Index: ProvName}
        i2p = {int(k) if str(k).isdigit() else k: v for k, v in class_map.items()}
        print(f" ✅ Using EMBEDDED class map from checkpoint ({len(class_map)} classes)")
    else:
        print(" ❌ Error: Province checkpoint must contain 'class_map'")
        return

    # ดึง Weights จากห่อ
    state_dict = prov_ckpt.get("model_state", prov_ckpt.get("model", prov_ckpt))
    prov_model = ProvinceClassifier(n_classes=len(class_map)).to(cfg.DEVICE)
    prov_model.load_state_dict(state_dict)
    prov_model.eval()
    print(" ✅ Province Model Weights Loaded")

    # 3. Load Test Data
    if not cfg.TEST_CSV.exists():
        print(f"❌ Test CSV not found: {cfg.TEST_CSV}")
        return
        
    test_df = pd.read_csv(cfg.TEST_CSV).fillna("")
    test_df = test_df[test_df['image'] != ""]
    print(f"Found {len(test_df)} test samples in CSV")

    # Transforms
    tf_ocr = get_ocr_transforms(is_train=False)
    tf_prov = get_prov_transforms(is_train=False)

    results = []
    
    # 4. Evaluation Loop
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            ocr_rel_path = row["image"]
            ocr_full_path = cfg.CROPS_DIR / ocr_rel_path
            
            prov_rel_path = ocr_rel_path.replace("/plates/", "/provs/").replace("_plate", "_prov")
            prov_full_path = cfg.CROPS_DIR / prov_rel_path

            gt_plate = str(row["gt_plate"]).strip()
            gt_prov = str(row["gt_province"]).strip()

            # Predict OCR
            pred_plate = ""
            if ocr_full_path.exists():
                try:
                    img = Image.open(ocr_full_path).convert("RGB")
                    t_img = tf_ocr(img).unsqueeze(0).to(cfg.DEVICE)
                    output = ocr_model(t_img)
                    pred_plate = best_path_decode(output.softmax(-1), int_to_char)[0]
                except Exception as e:
                    pass
            
            # Predict Province
            pred_prov = "Unknown"
            if prov_full_path.exists():
                try:
                    img = Image.open(prov_full_path).convert("RGB")
                    t_img = tf_prov(img).unsqueeze(0).to(cfg.DEVICE)
                    output = prov_model(t_img)
                    idx = output.argmax(1).item()
                    pred_prov = i2p.get(idx, "Unknown")
                except Exception as e:
                    pass

            # Metrics
            cer = editdistance.eval(pred_plate, gt_plate) / max(1, len(gt_plate))
            prov_acc = 1 if pred_prov == gt_prov else 0
            
            results.append({
                "image": ocr_rel_path,
                "gt_plate": gt_plate, "pred_plate": pred_plate, "cer": cer,
                "gt_prov": gt_prov, "pred_prov": pred_prov, "prov_acc": prov_acc
            })

    # 5. Summary
    if results:
        df_res = pd.DataFrame(results)
        print("\n" + "="*30)
        print(f"FINAL TEST RESULTS")
        print(f"OCR Avg CER:      {df_res['cer'].mean():.4f}")
        print(f"Prov Accuracy:    {df_res['prov_acc'].mean()*100:.2f}%")
        print("="*30)
        out_csv = cfg.PROJECT_ROOT / "test_evaluation_results.csv"
        df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"Details: {out_csv}")

if __name__ == "__main__":
    main()