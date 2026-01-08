import torch 
import pandas as pd
import numpy as np
import json
import editdistance
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image

# Import Local Modules
from src.config import cfg
from src.models import ResNetCRNN, ProvinceClassifier, best_path_decode
from src.preprocess import get_ocr_transforms, get_prov_transforms

class LPREvaluator:
    def __init__(self, device=cfg.DEVICE):
        print("\n=== LPR Evaluator Initializing ===")
        self.device = device
        
        self._load_mappings()
        
        self._setup_models()
        
        self.tf_ocr = get_ocr_transforms(is_train=False)
        self.tf_prov = get_prov_transforms(is_train=False)

    def _load_mappings(self):
        # OCR Mapping
        if not cfg.CHAR_MAP_PATH.exists():
            raise FileNotFoundError(f"OCR char_map not found at {cfg.CHAR_MAP_PATH}")
        with open(cfg.CHAR_MAP_PATH, 'r', encoding='utf-8') as f:
            self.int_to_char = json.load(f)
            
        # Province Mapping
        if not cfg.PROV_MAP_PATH.exists():
            raise FileNotFoundError(f"Province map not found at {cfg.PROV_MAP_PATH}")
        with open(cfg.PROV_MAP_PATH, 'r', encoding='utf-8') as f:
            self.int_to_prov = json.load(f)

    def _setup_models(self):
        # OCR Model
        self.ocr_model = ResNetCRNN(img_channel=1, num_classes=len(self.int_to_char)).to(self.device)
        if cfg.OCR_MODEL_SAVE_PATH.exists():
            ckpt = torch.load(cfg.OCR_MODEL_SAVE_PATH, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
            self.ocr_model.load_state_dict(state_dict)
            print(f" Loaded OCR Weights from {cfg.OCR_MODEL_SAVE_PATH.name}")
        self.ocr_model.eval()

        # Province Model
        self.prov_model = ProvinceClassifier(n_classes=len(self.int_to_prov)).to(self.device)
        if cfg.PROV_MODEL_SAVE_PATH.exists():
            ckpt = torch.load(cfg.PROV_MODEL_SAVE_PATH, map_location=self.device)
            state_dict = ckpt.get("model_state", ckpt.get("model", ckpt))
            self.prov_model.load_state_dict(state_dict)
            print(f" Loaded Province Weights from {cfg.PROV_MODEL_SAVE_PATH.name}")
        self.prov_model.eval()

    def predict_ocr(self, img_path):
        if not Path(img_path).exists(): return None
        try:
            img = Image.open(img_path).convert("RGB")
            t_img = self.tf_ocr(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.ocr_model(t_img)
                return best_path_decode(output.softmax(-1), self.int_to_char)[0]
        except: return None

    def predict_province(self, img_path):
        if not Path(img_path).exists(): return None
        try:
            img = Image.open(img_path).convert("RGB")
            t_img = self.tf_prov(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.prov_model(t_img)
                idx = output.argmax(1).item()
                return self.int_to_prov.get(str(idx), self.int_to_prov.get(idx, ""))
        except: return None

    def run(self, csv_path, crops_dir):
        print(f"\nEvaluating: {csv_path}")
        df = pd.read_csv(csv_path).fillna("")
        results, missing_count = [], 0
        
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Testing"):
            p_plate = self.predict_ocr(Path(crops_dir) / row.image)
            p_prov = self.predict_province(Path(crops_dir) / row.image.replace("/plates/", "/provs/").replace("_plate", "_prov"))
            
            if p_plate is None or p_prov is None:
                missing_count += 1
            
            # Simple fallback to empty string for both
            res_plate, res_prov = p_plate or "", p_prov or ""
            gt_plate, gt_prov = str(row.gt_plate).strip(), str(row.gt_province).strip()
            
            results.append({
                "image": row.image,
                "gt_plate": gt_plate, "pred_plate": res_plate,
                "gt_prov": gt_prov, "pred_prov": res_prov,
                "cer": editdistance.eval(res_plate, gt_plate) / max(1, len(gt_plate)),
                "prov_acc": 1 if res_prov == gt_prov else 0
            })
        
        res_df = pd.DataFrame(results)
        print(f"\nResults: Samples={len(res_df)}, Missing={missing_count}")
        print(f"OCR CER: {res_df['cer'].mean():.4f}, Prov Acc: {res_df['prov_acc'].mean()*100:.2f}%")
        return res_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPR Evaluation Script")
    parser.add_argument("--csv", type=str, default=str(cfg.TEST_CSV), help="Path to test CSV")
    parser.add_argument("--crops", type=str, default=str(cfg.CROPS_DIR), help="Path to crops directory")
    parser.add_argument("--output", type=str, default="test_results.csv", help="Output results filename")
    args = parser.parse_args()

    # 1. Initialize Evaluator
    evaluator = LPREvaluator()

    # 2. Run Evaluation using paths from Args (defaulting to Config)
    results_df = evaluator.run(args.csv, args.crops)

    # 3. Save Results
    output_path = cfg.PROJECT_ROOT / args.output
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Full details saved to: {output_path}")