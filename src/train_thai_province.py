"""
src/train_thai_province.py

Trains Model 3B (MobileNetV2 Thai Province Classifier) on 77 Thai provinces
using the newly extracted 3,739 crops.

Features:
  - Preserves aspect ratio via SmartResize((224, 224))
  - Class-weighted Cross-Entropy Loss to balance rare provinces against Bangkok
  - Real-time Top-1 and Top-5 accuracy evaluation
  - Drop-in replacement state_dict compatible with weights/province_map.json
  - Evaluates on independent test set upon training completion
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
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.models import ResNetProvinceClassifier
from src.preprocess import get_prov_transforms

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "thai_province_crops"
PROV_MAP_PATH = PROJECT_ROOT / "weights" / "province_map.json"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MODEL_SAVE_PATH = WEIGHTS_DIR / "province_model.pth"
BACKUP_SAVE_PATH = WEIGHTS_DIR / "province_model_backup.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


class ThaiProvinceDataset(Dataset):
    def __init__(self, split_dir, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        self.split_dir = Path(split_dir)

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


def compute_class_weights(samples, n_classes=77):
    counts = np.zeros(n_classes, dtype=np.float32)
    for _, label in samples:
        counts[label] += 1
    total = float(len(samples))
    # Smoothed inverse frequency (square root smoothing prevents exploding weights)
    weights = (total / (n_classes * np.maximum(counts, 1.0))) ** 0.5
    weights = np.clip(weights, 0.2, 8.0)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, top_k=5):
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


def train_thai_province(epochs=20, batch_size=32, lr=2e-4):
    print(f"\n=======================================================")
    print(f"--- Training Model 3B (Thai Province Classifier) ---")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_DIR}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    print(f"=======================================================\n")

    # 1. Load province map
    with open(PROV_MAP_PATH, "r", encoding="utf-8") as f:
        prov_map = json.load(f)
    n_classes = len(prov_map)
    print(f"Loaded province_map: {n_classes} classes.")

    # Build folder-to-idx mapping (e.g. '00_กระบี่' -> 0)
    class_to_idx = {}
    for k, v in prov_map.items():
        folder_name = f"{int(k):02d}_{v}"
        class_to_idx[folder_name] = int(k)

    # 2. Transforms
    train_tf = get_prov_transforms(is_train=True)
    val_tf = get_prov_transforms(is_train=False)

    # 3. Datasets
    train_ds = ThaiProvinceDataset(DATA_DIR / "train", class_to_idx, transform=train_tf)
    val_ds = ThaiProvinceDataset(DATA_DIR / "valid", class_to_idx, transform=val_tf)
    test_ds = ThaiProvinceDataset(DATA_DIR / "test", class_to_idx, transform=val_tf)

    print(f"Loaded dataset: Train={len(train_ds)}, Valid={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 4. Model Architecture: ResNetProvinceClassifier (ResNet34 with 77 outputs)
    model = ResNetProvinceClassifier(n_classes=n_classes, backbone="resnet34", pretrained=True)
    model = model.to(DEVICE)

    # 5. Loss with class weights
    class_weights = compute_class_weights(train_ds.samples, n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Backup existing checkpoint
    if MODEL_SAVE_PATH.exists():
        shutil.copy2(MODEL_SAVE_PATH, BACKUP_SAVE_PATH)
        print(f"Backed up previous checkpoint to: {BACKUP_SAVE_PATH.name}")

    best_val_top1 = 0.0
    best_val_top5 = 0.0
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

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_correct/train_total*100:.1f}%"})

        scheduler.step()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, DEVICE, top_k=5)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Top-1: {val_top1*100:.2f}%, Top-5: {val_top5*100:.2f}%"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_val_top5 = val_top5
            best_epoch = epoch

            torch.save({
                "model_state": model.state_dict(),
                "class_map": prov_map,
                "backbone": "resnet34",
                "best_acc": best_val_top1,
                "best_acc_top5": best_val_top5,
                "epoch": best_epoch,
            }, str(MODEL_SAVE_PATH))
            print(f"   --> New Best Checkpoint saved! (Val Top-1: {val_top1*100:.2f}%, Top-5: {val_top5*100:.2f}%)")

    print(f"\n=======================================================")
    print(f"Training Complete!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Top-1 Accuracy: {best_val_top1*100:.2f}%")
    print(f"Best Validation Top-5 Accuracy: {best_val_top5*100:.2f}%")
    print(f"Saved best model to: {MODEL_SAVE_PATH}")
    print(f"=======================================================\n")

    # Final Evaluation on Independent Test Set
    print("--- Evaluating Best Checkpoint on Test Set ---")
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_top1, test_top5 = evaluate(model, test_loader, DEVICE, top_k=5)
    print(f"Test Set Top-1 Accuracy: {test_top1*100:.2f}%")
    print(f"Test Set Top-5 Accuracy: {test_top5*100:.2f}%")
    print(f"Test Set Loss: {test_loss:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train_thai_province(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
