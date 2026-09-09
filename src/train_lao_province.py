"""
src/train_lao_province.py

Trains Model 3B_Lao (ResNet18 Lao Province Classifier) on 18 Lao Provinces
using the newly extracted Lao province crops with rectangular aspect ratio (256x64).

Features:
  - Architecture: ResNet18 with AdaptiveAvgPool2d((1, 1))
  - Rectangular input: SmartResize((256, 64), mode="RGB")
  - Balanced training with augmented minority classes
  - Inverse frequency class-weighted CrossEntropy loss
  - Top-1 and Top-3 accuracy evaluation on validation & test splits
  - Saves best checkpoint to weights/province_model_lao.pth
"""

import os
import shutil
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from src.models import ResNetProvinceClassifier
from src.preprocess import get_prov_transforms

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DATA_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao_province_crops"
PROV_MAP_PATH = PROJECT_ROOT / "weights" / "province_map_lao.json"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MODEL_SAVE_PATH = WEIGHTS_DIR / "province_model_lao.pth"
BACKUP_SAVE_PATH = WEIGHTS_DIR / "province_model_lao_backup.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


class LaoProvinceCropDataset(Dataset):
    def __init__(self, split_dir, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        self.split_dir = Path(split_dir)

        if not self.split_dir.exists():
            return

        for folder in sorted(self.split_dir.iterdir()):
            if folder.is_dir() and folder.name in class_to_idx:
                label = class_to_idx[folder.name]
                for img_p in folder.glob("*.jpg"):
                    self.samples.append((img_p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, label = self.samples[idx]
        img = Image.open(img_p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def compute_class_weights(samples, n_classes=18):
    counts = np.zeros(n_classes, dtype=np.float32)
    for _, label in samples:
        counts[label] += 1
    total = float(len(samples))
    weights = total / (n_classes * (counts + 1e-5))
    weights = np.clip(weights, 0.2, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, top_k=3):
    model.eval()
    val_loss = 0.0
    val_correct_1 = 0
    val_correct_k = 0
    val_total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * len(labels)
            preds1 = outputs.argmax(dim=1)
            val_correct_1 += (preds1 == labels).sum().item()

            _, topk = outputs.topk(min(top_k, outputs.size(1)), dim=1)
            val_correct_k += sum([labels[i].item() in topk[i] for i in range(len(labels))])
            val_total += len(labels)

    if val_total == 0:
        return 0.0, 0.0, 0.0

    return val_loss / val_total, val_correct_1 / val_total, val_correct_k / val_total


def train_lao_province_model(epochs=15, batch_size=32, lr=2e-4):
    print(f"\n=======================================================")
    print(f"--- Training Model 3B_Lao (ResNet18 Lao Province Classifier) ---")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_DIR}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    print(f"=======================================================\n")

    with open(PROV_MAP_PATH, "r", encoding="utf-8") as f:
        prov_map = json.load(f)
    n_classes = len(prov_map)
    print(f"Loaded Lao province map: {n_classes} classes.")

    class_to_idx = {}
    for k, v in prov_map.items():
        folder_name = f"{int(k):02d}_{v}"
        class_to_idx[folder_name] = int(k)

    train_tf = get_prov_transforms(is_train=True)
    val_tf = get_prov_transforms(is_train=False)

    train_ds = LaoProvinceCropDataset(DATA_DIR / "train", class_to_idx, transform=train_tf)
    val_ds = LaoProvinceCropDataset(DATA_DIR / "valid", class_to_idx, transform=val_tf)
    test_ds = LaoProvinceCropDataset(DATA_DIR / "test", class_to_idx, transform=val_tf)

    print(f"Loaded dataset: Train={len(train_ds)}, Valid={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Backbone: ResNet18
    model = ResNetProvinceClassifier(n_classes=n_classes, backbone="resnet18", pretrained=True)
    model = model.to(DEVICE)

    class_weights = compute_class_weights(train_ds.samples, n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Backup existing checkpoint
    if MODEL_SAVE_PATH.exists():
        shutil.copy2(MODEL_SAVE_PATH, BACKUP_SAVE_PATH)
        print(f"Backed up previous checkpoint to: {BACKUP_SAVE_PATH.name}")

    best_val_top1 = 0.0
    best_val_top3 = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_correct / train_total * 100:.1f}%"})

        scheduler.step()

        val_loss, val_top1, val_top3 = evaluate(model, val_loader, DEVICE, top_k=3)
        train_acc = train_correct / train_total if train_total > 0 else 0
        print(
            f"Epoch {epoch:02d}/{epochs:02d} -> "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Top-1: {val_top1*100:.2f}% | "
            f"Val Top-3: {val_top3*100:.2f}%"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_val_top3 = val_top3
            best_epoch = epoch

            torch.save({
                "model_state": model.state_dict(),
                "class_map": prov_map,
                "backbone": "resnet18",
                "best_acc": best_val_top1,
                "best_acc_top3": best_val_top3,
                "epoch": best_epoch,
            }, str(MODEL_SAVE_PATH))
            print(f"   --> New Best Checkpoint saved! (Val Top-1: {val_top1*100:.2f}%, Top-3: {val_top3*100:.2f}%)")

    print(f"\n=======================================================")
    print(f"Training Complete!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Top-1: {best_val_top1*100:.2f}%")
    print(f"Best Validation Top-3: {best_val_top3*100:.2f}%")
    print(f"Saved best model to: {MODEL_SAVE_PATH}")
    print(f"=======================================================\n")

    # Final Evaluation on Independent Test Set
    print("--- Evaluating Best Checkpoint on Test Set ---")
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_top1, test_top3 = evaluate(model, test_loader, DEVICE, top_k=3)
    print(f"Test Set Top-1 Accuracy: {test_top1*100:.2f}%")
    print(f"Test Set Top-3 Accuracy: {test_top3*100:.2f}%")
    print(f"Test Set Loss: {test_loss:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train_lao_province_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
