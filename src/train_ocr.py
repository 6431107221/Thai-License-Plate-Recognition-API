"""
src/train_ocr.py

Trains Model 3:
1. Model 3A (OCRTrainer): ResNet18 + BiLSTM + CTC loss for Thai license plate characters.
2. Model 3B (ProvinceTrainer): MobileNetV2 for 77 Thai province classification.
"""

import argparse
import json
import os
from pathlib import Path
import sys

# Enable CPU fallback for operators not yet implemented in MPS (e.g. aten::_ctc_loss)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.datasets import OCRDataset, ProvinceDataset, ocr_collate
from src.models import ProvinceClassifier, ResNetCRNN, best_path_decode
from src.preprocess import get_ocr_transforms, get_prov_transforms
from src.validators import format_thai_plate, is_valid_plate


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates Levenshtein edit distance between two strings without external dependencies."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_macro_f1(all_labels: np.ndarray, all_preds: np.ndarray, num_classes: int) -> float:
    """Computes Macro F1 score across present classes in numpy."""
    f1s = []
    for c in range(num_classes):
        tp = np.sum((all_preds == c) & (all_labels == c))
        fp = np.sum((all_preds == c) & (all_labels != c))
        fn = np.sum((all_preds != c) & (all_labels == c))
        if tp + fp + fn > 0:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


# ==========================================
# 1. Province Trainer (MobileNetV2)
# ==========================================
class ProvinceTrainer:
    def __init__(self):
        print("\n=== Province Trainer (MobileNetV2) ===")
        self.device = cfg.DEVICE

        if not cfg.PROV_MAP_PATH.exists():
            raise FileNotFoundError(f"Province map not found at {cfg.PROV_MAP_PATH}. Please prepare it first.")
        with open(cfg.PROV_MAP_PATH, "r", encoding="utf-8") as f:
            self.int_to_char = json.load(f)
        self.char_to_int = {v: int(k) for k, v in self.int_to_char.items()}

        self._prepare_data()
        self.best_acc = 0.0
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
        valid_rows = []
        for row in df.itertuples(index=False):
            gt = getattr(row, "gt_province", "")
            if not str(gt).strip():
                valid_rows.append(False)
                continue
            rel_p = getattr(row, "province_image", getattr(row, "image", ""))
            valid_rows.append((cfg.CROPS_DIR / str(rel_p)).exists())
        return df[valid_rows]

    def _setup_model(self):
        ckpt = None
        if cfg.PROV_MODEL_SAVE_PATH.exists():
            print(f" Found existing checkpoint: {cfg.PROV_MODEL_SAVE_PATH}")
            try:
                ckpt = torch.load(cfg.PROV_MODEL_SAVE_PATH, map_location=self.device)
            except Exception as e:
                print(f" Could not read checkpoint: {e}")

        # Init Datasets
        self.train_ds_real = ProvinceDataset(
            self.train_df, cfg.CROPS_DIR, char_map=self.char_to_int, transform=get_prov_transforms(is_train=True)
        )
        self.train_ds = self.train_ds_real
        self.val_ds = ProvinceDataset(
            self.val_df, cfg.CROPS_DIR, char_map=self.char_to_int, transform=get_prov_transforms(is_train=False)
        )

        # Init Loaders
        is_cuda = (self.device.type == "cuda")
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.BATCH_SIZE_PROV,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=is_cuda,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.BATCH_SIZE_PROV,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=is_cuda,
        )

        # Init Model (n_classes = len of province map, e.g. 77)
        self.model = ProvinceClassifier(len(self.int_to_char)).to(self.device)

        # Load Weights
        if ckpt:
            try:
                state_dict = ckpt.get("model_state", ckpt)
                new_state = {f"model.{k}" if not k.startswith("model.") else k: v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state, strict=False)
                if "best_acc" in ckpt:
                    self.best_acc = ckpt["best_acc"]
                print(f" Model weights loaded. Resuming Best Acc: {self.best_acc:.4f}")
            except Exception as e:
                print(f" Failed to load weights: {e}")

    def _setup_weights(self):
        master_map = self.train_ds_real.char_map
        all_labels = [master_map.get(str(getattr(r, "gt_province", "")).strip(), 0) for r in self.train_df.itertuples(index=False)]
        class_counts = np.bincount(all_labels, minlength=len(master_map))
        class_counts = np.where(class_counts == 0, 1, class_counts)

        weights = len(all_labels) / (len(master_map) * class_counts)
        weights = np.clip(weights, 0.1, 10.0)
        self.class_weights = torch.FloatTensor(weights).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", patience=4, factor=0.5)

    def train(self):
        print(f"Start Training Province for {cfg.EPOCHS} epochs...")
        patience = 0

        for ep in range(self.start_epoch, cfg.EPOCHS):
            self.model.train()
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
                pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/max(total, 1):.2%}")

            # Validation
            val_acc = self.validate()
            self.scheduler.step(val_acc)

            # Checkpoint
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                patience = 0
                self.save_checkpoint(ep, val_acc)
                print(f"   └── New Best Accuracy: {val_acc:.2%}")
            else:
                patience += 1
                print(f"   └── Val Acc: {val_acc:.2%} (Best: {self.best_acc:.2%}) [Patience: {patience}/{cfg.EARLY_STOPPING_PATIENCE}]")
                if patience >= cfg.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered for Province Trainer.")
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

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = float(np.mean(all_preds == all_labels)) if len(all_labels) > 0 else 0.0
        f1 = compute_macro_f1(all_labels, all_preds, len(self.int_to_char))
        print(f"       [Val] Accuracy: {acc:.2%}, Macro F1: {f1:.4f}")
        return acc

    def save_checkpoint(self, epoch, acc):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "class_map": self.int_to_char,
                "best_acc": acc,
                "epoch": epoch,
            },
            cfg.PROV_MODEL_SAVE_PATH,
        )
        print("       Province Model Saved!")


# ==========================================
# 2. OCR Trainer (ResNet18 + BiLSTM + CTC)
# ==========================================
class OCRTrainer:
    def __init__(self):
        print("\n=== OCR Trainer (ResNetCRNN + CTC) ===")
        self.device = cfg.DEVICE

        if not cfg.CHAR_MAP_PATH.exists():
            raise FileNotFoundError(f"OCR char_map not found at {cfg.CHAR_MAP_PATH}. Please prepare it first.")
        with open(cfg.CHAR_MAP_PATH, "r", encoding="utf-8") as f:
            self.int_to_char = json.load(f)
        self.char_to_int = {v: int(k) for k, v in self.int_to_char.items()}

        self._prepare_data()
        self.best_cer = 1.0
        self.best_score = 100.0  # Score = CER + (1 - FmtAcc)
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
        valid_rows = []
        for row in df.itertuples(index=False):
            gt = getattr(row, "gt_plate", "")
            if not str(gt).strip():
                valid_rows.append(False)
                continue
            rel_p = getattr(row, "plate_image", getattr(row, "image", ""))
            valid_rows.append((cfg.CROPS_DIR / str(rel_p)).exists())
        return df[valid_rows]

    def _setup_model(self):
        self.train_ds = OCRDataset(
            self.train_df, cfg.CROPS_DIR, char_map=self.char_to_int, transform=get_ocr_transforms(True)
        )
        self.val_ds = OCRDataset(
            self.val_df, cfg.CROPS_DIR, char_map=self.char_to_int, transform=get_ocr_transforms(False)
        )

        is_cuda = (self.device.type == "cuda")
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.BATCH_SIZE_OCR,
            shuffle=True,
            collate_fn=ocr_collate,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=is_cuda,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.BATCH_SIZE_OCR,
            collate_fn=ocr_collate,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=is_cuda,
        )

        self.model = ResNetCRNN(1, len(self.int_to_char), hidden_size=256).to(self.device)

        if cfg.OCR_MODEL_SAVE_PATH.exists():
            print(f" Resuming from {cfg.OCR_MODEL_SAVE_PATH}")
            try:
                ckpt = torch.load(cfg.OCR_MODEL_SAVE_PATH, map_location=self.device)
                state_dict = ckpt.get("model_state_dict", ckpt)
                self.model.load_state_dict(state_dict, strict=False)
                if "cer" in ckpt:
                    self.best_cer = ckpt["cer"]
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

            for batch in pbar:
                imgs, tg, tg_lens, _, _, _ = batch
                imgs = imgs.to(self.device)
                tg = tg.to(self.device)
                tg_lens = tg_lens.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(imgs)
                logp = out.log_softmax(-1)
                logp_loss = logp.permute(1, 0, 2)

                input_lengths = torch.full((imgs.size(0),), out.size(1), dtype=torch.long).to(self.device)
                loss = self.criterion(logp_loss, tg, input_lengths, tg_lens)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Validation
            val_cer, val_fmt_acc = self.validate()
            combined_score = val_cer + (1.0 - val_fmt_acc)
            self.scheduler.step(combined_score)

            if combined_score < self.best_score:
                self.best_score = combined_score
                self.best_cer = val_cer
                patience = 0
                self.save_checkpoint(ep, val_cer, val_fmt_acc)
                print(f"   └── Score: {combined_score:.4f} (CER: {val_cer:.4f}, Fmt: {val_fmt_acc:.2%}) -> New Best!")
            else:
                patience += 1
                print(f"   └── Score: {combined_score:.4f} (Best: {self.best_score:.4f}) [Patience: {patience}/{cfg.EARLY_STOPPING_PATIENCE}]")
                if patience >= cfg.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered for OCR Trainer.")
                    break

    def validate(self):
        self.model.eval()
        cer_sum = 0
        tot = 0
        valid_format_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                imgs, _, _, texts, _, _ = batch
                imgs = imgs.to(self.device)
                out = self.model(imgs)

                preds = best_path_decode(out.softmax(-1), self.int_to_char)

                for i, gt in enumerate(texts):
                    pred_raw = preds[i]
                    pred_fmt = format_thai_plate(pred_raw)

                    if is_valid_plate(pred_fmt):
                        valid_format_count += 1

                    div = max(1, len(gt))
                    base_cer = levenshtein_distance(pred_fmt, gt) / div
                    cer_sum += min(1.0, base_cer)
                    tot += 1

        avg_cer = cer_sum / max(1, tot)
        fmt_acc = valid_format_count / max(1, tot)
        print(f"       [Val] Format Valid Rate: {fmt_acc:.2%} ({valid_format_count}/{tot}), CER: {avg_cer:.4f}")
        return avg_cer, fmt_acc

    def save_checkpoint(self, epoch, cer, fmt_acc):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "int_to_char": self.int_to_char,
                "epoch": epoch,
                "cer": cer,
                "fmt_acc": fmt_acc,
                "score": self.best_score,
            },
            cfg.OCR_MODEL_SAVE_PATH,
        )
        print("       OCR Model Saved!")


# ==========================================
# Main CLI
# ==========================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Model 3 (OCR and Province Classifier)")
    p.add_argument("--task", type=str, default="all", choices=["ocr", "province", "all"], help="Task to train")
    p.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    p.add_argument("--batch", type=int, default=32, help="Batch size")
    args = p.parse_args()

    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE_OCR = args.batch
    cfg.BATCH_SIZE_PROV = args.batch
    cfg.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.task in ["province", "all"]:
        prov_trainer = ProvinceTrainer()
        prov_trainer.train()

    if args.task in ["ocr", "all"]:
        ocr_trainer = OCRTrainer()
        ocr_trainer.train()