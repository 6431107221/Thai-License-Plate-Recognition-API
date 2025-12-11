import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import json
import editdistance
from sklearn.metrics import f1_score
from pathlib import Path

# Import
from config import cfg
from models import ResNetCRNN, ProvinceClassifier
from datasets import OCRDataset, ProvinceDataset, ocr_collate, get_ocr_transforms, get_prov_transforms
from utils import best_path_decode

class ProvinceTrainer:
    def __init__(self):
        print("\n=== Province Trainer ===")
        self.device = cfg.DEVICE
        
        self._prepare_data()
        
        self._setup_model()
        
        self._setup_optimization()
        
        self.best_f1 = 0.0
        self.start_epoch = 0

    def _prepare_data(self):
        if not cfg.TRAIN_CSV.exists():
            raise FileNotFoundError(f"CSV not found at {cfg.TRAIN_CSV}")
            
        train_df_raw = pd.read_csv(cfg.TRAIN_CSV).fillna("")
        val_df_raw = pd.read_csv(cfg.VAL_CSV).fillna("")
        
        self.train_df = self._filter_existing(train_df_raw)
        self.val_df = self._filter_existing(val_df_raw)
        print(f" Train samples: {len(self.train_df)}, Val samples: {len(self.val_df)}")

    def _filter_existing(self, df):
        valid_rows = []
        for _, row in df.iterrows():
            img_rel_plate = row["image"]
            img_rel_prov = img_rel_plate.replace("/plates/", "/provs/").replace("__plate", "__prov")
            if (cfg.CROPS_DIR / img_rel_prov).exists():
                valid_rows.append(row)
        return pd.DataFrame(valid_rows)

    def _setup_model(self):
        forced_map = None
        load_source = None
        
        if cfg.PROV_MODEL_SAVE_PATH.exists():
            load_source = cfg.PROV_MODEL_SAVE_PATH
            print(f" Found existing checkpoint: {load_source}")
        elif cfg.PROV_PRETRAINED_PATH.exists():
            load_source = cfg.PROV_PRETRAINED_PATH
            print(f" Found pretrained base: {load_source}")

        if load_source:
            try:
                ckpt = torch.load(load_source, map_location=self.device, weights_only=False)
                if "class_map" in ckpt:
                    forced_map = {v: int(k) for k, v in ckpt["class_map"].items()}
            except Exception as e:
                print(f" Warning: Could not read class map from checkpoint: {e}")

        # Init Datasets
        self.train_ds = ProvinceDataset(self.train_df, cfg.CROPS_DIR, class_map=forced_map, training=True)
        self.val_ds = ProvinceDataset(self.val_df, cfg.CROPS_DIR, class_map=self.train_ds.p2i, training=False)
        
        # Init Loaders
        is_cuda = (self.device.type == 'cuda')
        self.train_loader = DataLoader(self.train_ds, batch_size=cfg.BATCH_SIZE_PROV, shuffle=True, 
                                     num_workers=cfg.NUM_WORKERS, pin_memory=is_cuda)
        self.val_loader = DataLoader(self.val_ds, batch_size=cfg.BATCH_SIZE_PROV, shuffle=False, 
                                   num_workers=cfg.NUM_WORKERS, pin_memory=is_cuda)

        # Init Model
        self.model = ProvinceClassifier(len(self.train_ds.p2i)).to(self.device)
        
        # Load Weights
        if load_source:
            try:
                ckpt = torch.load(load_source, map_location=self.device, weights_only=False)
                state_dict = ckpt.get("model_state", ckpt)
                new_state = {f"model.{k}" if not k.startswith("model.") else k: v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state)
                if "best_f1" in ckpt: self.best_f1 = ckpt["best_f1"]
                if "epoch" in ckpt: self.start_epoch = ckpt["epoch"] + 1
                print(f" Model weights loaded. Resuming Best F1: {self.best_f1:.4f}")
            except Exception as e:
                print(f" Failed to load weights: {e}")

    def _setup_optimization(self):
        # Calculate Class Weights
        master_map = self.train_ds.p2i
        all_labels = [master_map.get(row["gt_province"], 0) for _, row in self.train_df.iterrows()]
        class_counts = np.bincount(all_labels, minlength=len(master_map))
        class_counts = np.where(class_counts == 0, 1, class_counts)
        
        weights = len(all_labels) / (len(master_map) * class_counts)
        weights = np.clip(weights, 1.0, 10.0)
        self.class_weights = torch.FloatTensor(weights).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=4, factor=0.5)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))

    def train(self):
        print(f"Start Training Province for {cfg.EPOCHS} epochs...")
        patience = 0
        
        for ep in range(self.start_epoch, cfg.EPOCHS):
            self.model.train()
            self.train_ds.training = True
            
            # Freeze BN
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()
            
            loss_sum = 0
            correct = 0
            total = 0
            
            pbar = tqdm(self.train_loader, desc=f"Prov Ep {ep+1}/{cfg.EPOCHS}")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    out = self.model(imgs)
                    loss = self.criterion(out, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
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
        self.val_ds.training = False
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs = imgs.to(self.device)
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    out = self.model(imgs)
                preds = out.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        return f1_score(all_labels, all_preds, average='macro')

    def save_checkpoint(self, epoch, f1):
        torch.save({
            "model_state": self.model.state_dict(),
            "class_map": self.train_ds.i2p,
            "best_f1": f1,
            "epoch": epoch
        }, cfg.PROV_MODEL_SAVE_PATH)
        print("       Model Saved!")


class OCRTrainer:
    def __init__(self):
        print("\n=== OCR Trainer ===")
        self.device = cfg.DEVICE
        
        # 1. Load Char Map
        with open(cfg.CHAR_MAP_PATH, 'r', encoding='utf-8') as f:
            self.int_to_char = json.load(f)
        self.char_to_int = {v: int(k) for k, v in self.int_to_char.items()}
        
        # 2. Datasets
        train_df = pd.read_csv(cfg.TRAIN_CSV).fillna("")
        val_df = pd.read_csv(cfg.VAL_CSV).fillna("")
        
        self.train_ds = OCRDataset(train_df, cfg.CROPS_DIR, self.char_to_int, transform=get_ocr_transforms(True))
        self.val_ds = OCRDataset(val_df, cfg.CROPS_DIR, self.char_to_int, transform=get_ocr_transforms(False))
        
        is_cuda = (self.device.type == 'cuda')
        self.train_loader = DataLoader(self.train_ds, batch_size=cfg.BATCH_SIZE_OCR, shuffle=True,
                                     collate_fn=ocr_collate, num_workers=cfg.NUM_WORKERS, pin_memory=is_cuda)
        self.val_loader = DataLoader(self.val_ds, batch_size=cfg.BATCH_SIZE_OCR, collate_fn=ocr_collate,
                                   num_workers=cfg.NUM_WORKERS, pin_memory=is_cuda)
        
        # 3. Model
        self.model = ResNetCRNN(1, len(self.int_to_char), hidden_size=256).to(self.device)
        self.best_cer = 1.0
        self.start_epoch = 1
        
        self._load_weights()
        self._setup_optimization()

    def _load_weights(self):
        load_source = None
        if cfg.OCR_MODEL_SAVE_PATH.exists():
            load_source = cfg.OCR_MODEL_SAVE_PATH
            print(f" Resuming from {load_source}")
        elif cfg.OCR_PRETRAINED_PATH.exists():
            load_source = cfg.OCR_PRETRAINED_PATH
            print(f" Loading pretrained base {load_source}")
            
        if load_source:
            try:
                ckpt = torch.load(load_source, map_location=self.device, weights_only=True)
                self.model.load_state_dict(ckpt["model_state_dict"])
                if "cer" in ckpt: self.best_cer = ckpt["cer"]
                if "epoch" in ckpt: self.start_epoch = ckpt["epoch"] + 1
                print(f" Weights loaded. Best CER: {self.best_cer:.4f}")
            except Exception as e:
                print(f" Failed to load weights: {e}")

    def _setup_optimization(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))

    def train(self):
        print(f"Start Training OCR for {cfg.EPOCHS} epochs...")
        
        for ep in range(self.start_epoch, cfg.EPOCHS + 1):
            self.model.train()
            total_loss = 0
            pbar = tqdm(self.train_loader, desc=f"OCR Ep {ep}")
            
            for batch in pbar:
                imgs, tg, tg_lens, _, _, _ = batch
                imgs, tg, tg_lens = imgs.to(self.device), tg.to(self.device), tg_lens.to(self.device)
                
                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    out = self.model(imgs)
                    logp = out.log_softmax(-1)
                    logp_loss = logp.permute(1, 0, 2)
                    input_lengths = torch.full((imgs.size(0),), out.size(1), dtype=torch.long).to(self.device)
                    loss = self.criterion(logp_loss, tg, input_lengths, tg_lens)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Validation
            val_cer = self.validate()
            print(f"   └── CER: {val_cer:.4f} (Best: {self.best_cer:.4f})")
            
            self.scheduler.step(val_cer)
            
            if val_cer < self.best_cer:
                self.best_cer = val_cer
                self.save_checkpoint(ep, val_cer)

    def validate(self):
        self.model.eval()
        cer_sum = 0
        tot = 0
        with torch.no_grad():
            for batch in self.val_loader:
                imgs, tg, tg_lens, _, _, _ = batch
                imgs = imgs.to(self.device)
                out = self.model(imgs)
                preds = best_path_decode(out, self.int_to_char)
                idx = 0
                for i, L in enumerate(tg_lens):
                    gt = "".join(self.int_to_char[str(x)] for x in tg[idx:idx+int(L)].tolist())
                    div = max(1, len(gt))
                    cer_sum += editdistance.eval(preds[i], gt) / div
                    tot += 1
                    idx += int(L)
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
    
    # 1. Train Province
    prov_trainer = ProvinceTrainer()
    prov_trainer.train()
    
    # 2. Train OCR
    ocr_trainer = OCRTrainer()
    ocr_trainer.train()