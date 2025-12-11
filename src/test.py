import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
import editdistance
from tqdm.auto import tqdm

# Import
from models import ResNetCRNN, ProvinceClassifier
from utils import beam_search_decode 
from datasets import get_ocr_transforms, get_prov_transforms

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CROPS_ROOT = Path("crops_all")
TEST_CSV_PATH = CROPS_ROOT / "test" / "test_unified.csv"

# Path
OCR_MODEL_PATH = Path("ocr_train_out/best_model.pth")
PROV_MODEL_PATH = Path("ocr_train_out/province_best.pth")
CHAR_MAP_PATH = Path("ocr_minimal/int_to_char.json")

# Transforms
tf_ocr_eval = get_ocr_transforms(is_train=False)
tf_prov_eval = get_prov_transforms(is_train=False)

def find_image_file(filename):
    if not filename: return None
    filename = str(filename).replace("\\", "/")
    
    candidates = [
        CROPS_ROOT / filename,
        CROPS_ROOT / Path(filename).name,
    ]
    for p in candidates:
        if p.exists(): return p
    return None

def main():
    # 1. Load Char Map
    if not CHAR_MAP_PATH.exists():
        print(f"Error: {CHAR_MAP_PATH} not found.")
        return

    with open(CHAR_MAP_PATH, 'r', encoding='utf-8') as f:
        int_to_char = json.load(f)
    
    # 2. Load Models
    # --- OCR Model ---
    ocr_model = ResNetCRNN(1, len(int_to_char), hidden_size=256, num_rnn_layers=2).to(DEVICE)
    
    if OCR_MODEL_PATH.exists():
        print(f"OCR model: {OCR_MODEL_PATH}...")
        try:
            ckpt = torch.load(OCR_MODEL_PATH, map_location=DEVICE, weights_only=True) 
            state_dict = ckpt["model_state_dict"]
            ocr_model.load_state_dict(state_dict) 
            ocr_model.eval() 
        except Exception as e:
            print(f"Load failed: {e}")
            return 
    else:
        print(f"Not found:{OCR_MODEL_PATH}")
        return

    # --- Province Model ---
    prov_idx2prov = {}
    
    if PROV_MODEL_PATH.exists():
        print(f"Province model: {PROV_MODEL_PATH}...")
        try:
            ckpt = torch.load(PROV_MODEL_PATH, map_location=DEVICE, weights_only=True)
            
            # ดึง Class Map
            if "class_map" in ckpt:
                prov_idx2prov = ckpt["class_map"]
                prov_idx2prov = {int(k):v for k,v in prov_idx2prov.items()}
            else:
                print("Class_map not found in province checkpoint.")
                return

            # Init Model
            prov_model = ProvinceClassifier(len(prov_idx2prov)).to(DEVICE)
            
            # ดึง State Dict
            state_dict = ckpt.get("model_state", ckpt)
            
            # แก้ไข Key prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith("model."):
                    new_state_dict[f"model.{k}"] = v
                else:
                    new_state_dict[k] = v
            
            prov_model.load_state_dict(new_state_dict)
            prov_model.eval()
            
        except Exception as e:
            print(f"Failed load Province model: {e}")
            return
    else:
        print(f"Province model not found: {PROV_MODEL_PATH}")
        return

    # 3. Load Test Data
    if not TEST_CSV_PATH.exists():
        print(f"Test CSV not found: {TEST_CSV_PATH}")
        return
        
    test_df = pd.read_csv(TEST_CSV_PATH, dtype=str).fillna("")
    print(f"Starting Inference: {len(test_df)} images")

    results = []
    
    # 4. Inference Loop
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            img_rel_path = row.get("image")
            img_path = find_image_file(img_rel_path)
            
            if img_path is None:
                continue

            # --- OCR Prediction ---
            pred_plate = ""
            try:
                pil_gray = Image.open(img_path).convert("L")
                
                ts_ocr = tf_ocr_eval(pil_gray).unsqueeze(0).to(DEVICE)
                
                out_ocr = ocr_model(ts_ocr)
                log_probs = out_ocr[0].log_softmax(-1)
                
                pred_plate = beam_search_decode(log_probs, int_to_char, beam_width=3)
            except Exception as e:
                print(f"OCR Error: {img_path.name}: {e}")

            # --- Province Prediction ---
            pred_prov = ""
            prov_name = img_path.name.replace("__plate", "__prov")
            prov_path = img_path.parent.parent / "provs" / prov_name
            
            if not prov_path.exists():
                 prov_path = find_image_file(prov_name)

            if prov_path and prov_path.exists():
                try:
                    pil_prov = Image.open(prov_path).convert("L")
                    
                    ts_prov = tf_prov_eval(pil_prov).unsqueeze(0).to(DEVICE)
                    
                    out_prov = prov_model(ts_prov)
                    idx = out_prov.argmax(1).item()
                    
                    pred_prov = prov_idx2prov.get(idx, str(idx))
                except Exception as e:
                    print(f"Province Error: {prov_name}: {e}")
            
            # --- Calculate Metrics ---
            gt_plate = row.get("gt_plate", "")
            gt_prov = row.get("gt_province", "")
            
            cer = 0.0
            if gt_plate:
                cer = editdistance.eval(pred_plate, gt_plate) / max(1, len(gt_plate))
            
            acc = 0
            if gt_prov:
                acc = 1 if pred_prov == gt_prov else 0

            results.append({
                "image": img_path.name,
                "gt_plate": gt_plate,
                "pred_plate": pred_plate,
                "cer": cer,
                "gt_province": gt_prov,
                "pred_province": pred_prov,
                "acc": acc
            })

    # 5. Save Results
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv("ocr_train_out/final_results.csv", index=False, encoding="utf-8-sig")
        
        avg_cer = res_df["cer"].mean()
        avg_acc = res_df["acc"].mean()
        
        print(f"Average CER: {avg_cer:.4f}")
        print(f"Province Accuracy: {avg_acc:.4%}")
    else:
        print("No results")

if __name__ == "__main__":
    main()