"""
src/train_character_classifier.py

Trains Model 3A-Box (MobileNetV2 Character Classifier) on the 50 Thai character classes
(10 numbers + 40 Thai consonants) using the balanced dataset in:
  datasets/thai_character_crops/splits/train/
  datasets/thai_character_crops/splits/valid/

Features:
  - Input: 64x64 RGB square-padded character crops
  - Class-weighted Cross-Entropy Loss to balance rare consonants against common digits
  - Real-time Top-1 and Top-3 accuracy evaluation on validation set
  - Saves best checkpoint to weights/character_classifier.pth
"""

import os
import sys
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

BASE_DIR = PROJECT_ROOT / "datasets" / "thai_character_crops"
SPLITS_DIR = BASE_DIR / "splits"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MAP_PATH = WEIGHTS_DIR / "char_classifier_map.json"
MODEL_SAVE_PATH = WEIGHTS_DIR / "character_classifier.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


class CharacterDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        self.root_dir = Path(root_dir)

        for folder in sorted(self.root_dir.iterdir()):
            if folder.is_dir() and folder.name in class_to_idx:
                label = class_to_idx[folder.name]
                for img_p in list(folder.glob("*.jpg")) + list(folder.glob("*.png")):
                    self.samples.append((img_p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, label = self.samples[idx]
        img = Image.open(img_p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def compute_class_weights(samples, n_classes=50):
    counts = np.zeros(n_classes, dtype=np.float32)
    for _, label in samples:
        counts[label] += 1
    total = float(len(samples))
    weights = (total / (n_classes * np.maximum(counts, 1.0))) ** 0.5
    weights = np.clip(weights, 0.2, 8.0)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    val_loss = 0.0
    val_correct_1 = 0
    val_correct_3 = 0
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

            _, top3 = outputs.topk(min(3, outputs.size(1)), dim=1)
            val_correct_3 += sum([labels[i].item() in top3[i] for i in range(len(labels))])
            val_total += len(labels)

    if val_total == 0:
        return 0.0, 0.0, 0.0

    return val_loss / val_total, val_correct_1 / val_total, val_correct_3 / val_total


def train_character_classifier(epochs=15, batch_size=64, lr=3e-4):
    print(f"\n=======================================================")
    print(f"--- Training Character Classifier (MobileNetV2) ---")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    print(f"=======================================================\n")

    # 1. Load class mapping
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        idx_to_char = json.load(f)
    n_classes = len(idx_to_char)
    class_to_idx = {v: int(k) for k, v in idx_to_char.items()}
    print(f"Total Character Classes: {n_classes}")

    # 2. Transforms (Input 64x64)
    train_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomAffine(degrees=5, translate=(0.04, 0.04)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Datasets
    train_ds = CharacterDataset(SPLITS_DIR / "train", class_to_idx, transform=train_tf)
    val_ds = CharacterDataset(SPLITS_DIR / "valid", class_to_idx, transform=val_tf)

    print(f"Loaded: Train = {len(train_ds)} samples, Valid = {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Model Architecture: MobileNetV2 with 50 outputs
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, n_classes)
    )
    model = model.to(DEVICE)

    # 5. Class weights & Optimizer
    class_weights = compute_class_weights(train_ds.samples, n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

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

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_correct/train_total*100:.1f}%"})

        scheduler.step()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation
        val_loss, val_top1, val_top3 = evaluate(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Top-1: {val_top1*100:.2f}%, Top-3: {val_top3*100:.2f}%"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_val_top3 = val_top3
            best_epoch = epoch

            torch.save({
                "model_state": model.state_dict(),
                "class_map": idx_to_char,
                "best_acc_top1": best_val_top1,
                "best_acc_top3": best_val_top3,
                "epoch": best_epoch,
            }, str(MODEL_SAVE_PATH))
            print(f"   --> New Best Checkpoint saved! (Val Top-1: {val_top1*100:.2f}%, Top-3: {val_top3*100:.2f}%)")

    print(f"\n=======================================================")
    print(f"Training Complete!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Top-1 Accuracy: {best_val_top1*100:.2f}%")
    print(f"Best Validation Top-3 Accuracy: {best_val_top3*100:.2f}%")
    print(f"Saved weights to: {MODEL_SAVE_PATH}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train_character_classifier(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
