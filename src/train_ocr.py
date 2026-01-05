import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

import editdistance
from sklearn.metrics import f1_score

import argparse
from tqdm.auto import tqdm
import json

# Import
from src.config import cfg
from src.models import ResNetCRNN, ProvinceClassifier, best_path_decode
from src.datasets import OCRDataset, ProvinceDataset, ocr_collate
from src.preprocess import get_ocr_transforms, get_prov_transforms

class ProvinceTrainer:
    def __init__(self):
        print("\n=== Province Trainer ===")
        self.device = cfg.DEVICE
        
        if not cfg.PROV_MAP_PATH.exists():
            raise FileNotFoundError(f"Province map not found at {cfg.PROV_MAP_PATH}. Please prepare it first.")
        with open(cfg.PROV_MAP_PATH, 'r', encoding='utf-8') as f:
            self.int_to_char = json.load(f)
        self.char_to_int = {v: int(k) for k, v in self.int_to_char.items()}

        self._prepare_data()
        self.best_f1 = 0.0
        self.start_epoch = 0
        
        self._setup_model()
        self._setup_weights()

    def _prepare_data(self):
        if not cfg.TRAIN_CSV.exists():
            raise FileNotFoundError(f"CSV not found at {cfg.TRAIN_CSV}")
            
        train_df_raw = pd.read_csv(cfg.TRAIN_CSV).fillna("")
        val_df_raw = pd.read_csv(cfg.VAL_CSV).fillna("")
        
        self.train_df = self._filter_existing(train_df_raw)
        self.val_df = self._filter_existing(val_df_raw)
        print(f" Train samples: {len(self.train_df)}, Val samples: {len(self.val_df)}")

    def _filter_existing(self, df):
        mask = [(cfg.CROPS_DIR / row.image.replace("/plates/", "/provs/").replace("_plate", "_prov")).exists() 
                for row in df.itertuples(index=False)]
        return df[mask]

    def _setup_model(self):
        ckpt = None
        if cfg.PROV_MODEL_SAVE_PATH.exists():
            print(f" Found existing checkpoint: {cfg.PROV_MODEL_SAVE_PATH}")
            try:
                ckpt = torch.load(cfg.PROV_MODEL_SAVE_PATH, map_location=self.device)
            except Exception as e:
                print(f" Could not read checkpoint: {e}")

        # Init Datasets
        self.train_ds_real = ProvinceDataset(self.train_df, cfg.CROPS_DIR, char_map=self.char_to_int, transform=get_prov_transforms(is_train=True))
        self.train_ds = self.train_ds_real 
        self.val_ds = ProvinceDataset(self.val_df, cfg.CROPS_DIR, char_map=self.char_to_int, transform=get_prov_transforms(is_train=False))
        
        # Init Loaders
        is_cuda = (self.device.type == 'cuda')
        self.train_loader = DataLoader(self.train_ds, batch_size=cfg.BATCH_SIZE_PROV, shuffle=True, 
                                     num_workers=cfg.NUM_WORKERS, pin_memory=is_cuda)
        self.val_loader = DataLoader(self.val_ds, batch_size=cfg.BATCH_SIZE_PROV, shuffle=False, 
                                   num_workers=cfg.NUM_WORKERS, pin_memory=is_cuda)

        # Init Model
        self.model = ProvinceClassifier(len(self.train_ds_real.char_map)).to(self.device)
        
        # Load Weights
        if ckpt:
            try:
                state_dict = ckpt.get("model_state", ckpt)
                new_state = {f"model.{k}" if not k.startswith("model.") else k: v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state, strict=False)
                if "best_f1" in ckpt: self.best_f1 = ckpt["best_f1"]
                print(f" Model weights loaded. Resuming Best F1: {self.best_f1:.4f}")
            except Exception as e:
                print(f" Failed to load weights: {e}")

    def _setup_weights(self):
        master_map = self.train_ds_real.char_map
        all_labels = [master_map.get(row.gt_province, 0) for row in self.train_df.itertuples(index=False)]
        class_counts = np.bincount(all_labels, minlength=len(master_map))
        class_counts = np.where(class_counts == 0, 1, class_counts)
        
        weights = len(all_labels) / (len(master_map) * class_counts)
        weights = np.clip(weights, 1.0, 10.0)
        self.class_weights = torch.FloatTensor(weights).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=4, factor=0.5)

    def train(self):
        print(f"Start Training Province for {cfg.EPOCHS} epochs...")
        patience = 0
        
        for ep in range(self.start_epoch, cfg.EPOCHS):
            self.model.train()

            
            # Freeze BN for stability on small batches
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()
            
            loss_sum = 0
            correct = 0
            total = 0
            
            pbar = tqdm(self.train_loader, desc=f"Prov Ep {ep+1}/{cfg.EPOCHS}")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                out = self.model(imgs)
                loss = self.criterion(out, labels)
                loss.backward()
                self.optimizer.step()
                
                loss_sum += loss.item()
                preds = out.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.2%}")
            
            # Validation
            val_f1 = self.validate()
            self.scheduler.step(val_f1)
            
            print(f"   └── Val F1: {val_f1:.4f} (Best: {self.best_f1:.4f})")
            
            # Checkpoint
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                patience = 0
                self.save_checkpoint(ep, val_f1)
            else:
                patience += 1
                if patience >= cfg.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

    def validate(self):
        self.model.eval()

        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs = imgs.to(self.device)
                out = self.model(imgs)
                preds = out.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        return f1_score(all_labels, all_preds, average='macro')

    def save_checkpoint(self, epoch, f1):
        torch.save({
            "model_state": self.model.state_dict(),
            "class_map": self.train_ds.int_to_char,
            "best_f1": f1,
            "epoch": epoch
        }, cfg.PROV_MODEL_SAVE_PATH)
        print("       Model Saved!")


class OCRTrainer:
    def __init__(self):
        print("\n=== OCR Trainer ===")
        self.device = cfg.DEVICE
        
        if not cfg.CHAR_MAP_PATH.exists():
            raise FileNotFoundError(f"OCR char_map not found at {cfg.CHAR_MAP_PATH}. Please prepare it first.")
        with open(cfg.CHAR_MAP_PATH, 'r', encoding='utf-8') as f:
            self.int_to_char = json.load(f)
        self.char_to_int = {v: int(k) for k, v in self.int_to_char.items()}
        
        self._prepare_data()
        self.best_cer = 1.0
        self.start_epoch = 0
        
        self._setup_model()
        self._setup_weight()

    def _prepare_data(self):
        train_df_raw = pd.read_csv(cfg.TRAIN_CSV).fillna("")
        val_df_raw = pd.read_csv(cfg.VAL_CSV).fillna("")
        
        self.train_df = self._filter_existing(train_df_raw)
        self.val_df = self._filter_existing(val_df_raw)
        print(f" Train samples: {len(self.train_df)}, Val samples: {len(self.val_df)}")
        
    def _filter_existing(self, df):
        mask = [(cfg.CROPS_DIR / row.image).exists() for row in df.itertuples(index=False)]
        return df[mask]

    def _setup_model(self):
        # Init Datasets
        self.train_ds_real = OCRDataset(self.train_df, cfg.CROPS_DIR, char_map=self.char_to_int, transform=get_ocr_transforms(True))
        self.train_ds = self.train_ds_real 
        self.val_ds = OCRDataset(self.val_df, cfg.CROPS_DIR, char_map=self.char_to_int, transform=get_ocr_transforms(False))
        
        # Init Loaders
        is_cuda = (self.device.type == 'cuda')
        self.train_loader = DataLoader(self.train_ds, batch_size=cfg.BATCH_SIZE_OCR, shuffle=True,
                                     collate_fn=ocr_collate, num_workers=cfg.NUM_WORKERS, pin_memory=is_cuda)
        self.val_loader = DataLoader(self.val_ds, batch_size=cfg.BATCH_SIZE_OCR, collate_fn=ocr_collate,
                                   num_workers=cfg.NUM_WORKERS, pin_memory=is_cuda)
        
        # Init Model
        self.model = ResNetCRNN(1, len(self.int_to_char), hidden_size=256).to(self.device)
        
        # Load Weights
        if cfg.OCR_MODEL_SAVE_PATH.exists():
            print(f" Resuming from {cfg.OCR_MODEL_SAVE_PATH}")
            try:
                ckpt = torch.load(cfg.OCR_MODEL_SAVE_PATH, map_location=self.device)
                state_dict = ckpt.get("model_state_dict", ckpt)
                self.model.load_state_dict(state_dict, strict=False)
                if "cer" in ckpt: self.best_cer = ckpt["cer"]
                print(f" Weights loaded. Best CER: {self.best_cer:.4f}")
            except Exception as e:
                print(f" Warning: Could not load weights: {e}")

    def _setup_weight(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)

    def train(self):
        print(f"Start Training OCR for {cfg.EPOCHS} epochs...")
        patience = 0
        
        for ep in range(self.start_epoch, cfg.EPOCHS):
            self.model.train()
            total_loss = 0
            pbar = tqdm(self.train_loader, desc=f"OCR Ep {ep+1}/{cfg.EPOCHS}")
            
            # Freeze BN for stability on small batches
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()
            
            for batch in pbar:
                imgs, tg, tg_lens, _, _, _ = batch
                imgs, tg, tg_lens = imgs.to(self.device), tg.to(self.device), tg_lens.to(self.device)
                
                self.optimizer.zero_grad()
                out = self.model(imgs)
                logp = out.log_softmax(-1)
                logp_loss = logp.permute(1, 0, 2)
                
                input_lengths = torch.full((imgs.size(0),), out.size(1), dtype=torch.long).to(self.device)
                loss = self.criterion(logp_loss, tg, input_lengths, tg_lens)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Clip 1.0 for stability
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Validation
            val_cer = self.validate()
            self.scheduler.step(val_cer)
            
            if val_cer < self.best_cer:
                self.best_cer = val_cer
                patience = 0
                self.save_checkpoint(ep, val_cer)
                print(f"   └── CER: {val_cer:.4f} (New Best!)")
            else:
                patience += 1
                print(f"   └── CER: {val_cer:.4f} (Best: {self.best_cer:.4f}) [Patience: {patience}/{cfg.EARLY_STOPPING_PATIENCE}]")
                if patience >= cfg.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

    def validate(self):
        self.model.eval()
        cer_sum = 0
        tot = 0
        with torch.no_grad():
            for batch in self.val_loader:
                imgs, _, _, texts, _, _ = batch
                imgs = imgs.to(self.device)
                out = self.model(imgs)
                
                # Decode using utility (implicit blank=0)
                preds = best_path_decode(out.softmax(-1), self.int_to_char) 
                
                for i, gt in enumerate(texts):
                    div = max(1, len(gt))
                    cer_sum += editdistance.eval(preds[i], gt) / div
                    tot += 1
        return cer_sum / max(1, tot)

    def save_checkpoint(self, epoch, cer):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "int_to_char": self.int_to_char,
            "epoch": epoch,
            "cer": cer
        }, cfg.OCR_MODEL_SAVE_PATH)
        print("       Model Saved!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="all", choices=["ocr", "province", "all"], help="Task to train (default: all)")
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS, help="Number of epochs")
    p.add_argument("--batch", type=int, default=cfg.BATCH_SIZE_OCR, help="Batch size")
    args = p.parse_args()
    
    # Update Config from Args
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE_OCR = args.batch
    cfg.BATCH_SIZE_PROV = args.batch
    
    # Ensure Weights Dir Exists
    cfg.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.task in ["province", "all"]:
        prov_trainer = ProvinceTrainer()
        prov_trainer.train()
        
    if args.task in ["ocr", "all"]:
        # Auto-generate Char Map if missing
        if not cfg.CHAR_MAP_PATH.exists():
            print(" Generating char map from training data...")
            df = pd.read_csv(cfg.TRAIN_CSV).fillna("")
            all_chars = sorted(list(set("".join(df["gt_plate"].astype(str).tolist()))))
            
            # Create Map (0 is reserved for CTC Blank)
            int_to_char = {str(i+1): c for i, c in enumerate(all_chars)}
            int_to_char["0"] = "<BLANK>" 
            
            with open(cfg.CHAR_MAP_PATH, "w", encoding="utf-8") as f:
                json.dump(int_to_char, f, ensure_ascii=False, indent=2)
            print(f" Saved char map to {cfg.CHAR_MAP_PATH} ({len(int_to_char)} tokens)")
            
        ocr_trainer = OCRTrainer()
        ocr_trainer.train()