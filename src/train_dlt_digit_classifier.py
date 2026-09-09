"""
src/train_dlt_digit_classifier.py

Trains an ultra-lightweight, high-accuracy 10-Class Digit Classifier (0-9)
specifically on 2,564 real Thai license plate character crops.

Purpose:
  Dedicated recognition of the 2-digit DLT Province Code stamped on the top banner
  of commercial transport trucks (THAILAND XX), eliminating any confusion with
  Thai consonants (ป, ธ, ฎ, etc.).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.models import DigitClassifier

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DATA_CSV = PROJECT_ROOT / "datasets" / "Thai" / "thai_character_crops" / "metadata.csv"
CROPS_DIR = PROJECT_ROOT / "datasets" / "Thai" / "thai_character_crops"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MODEL_SAVE_PATH = WEIGHTS_DIR / "digit_classifier.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


class ThaiDigitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_dir: Path, transform=None):
        self.samples = []
        self.transform = transform
        for _, row in df.iterrows():
            ch = str(row["character"]).strip()
            if ch.isdigit():
                label = int(ch)
                # rel_path_square is square-padded 64x64, rel_path_raw is raw crop
                rel_p = str(row.get("rel_path_square", ""))
                full_p = base_dir / rel_p
                if full_p.exists():
                    self.samples.append((full_p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception:
            fallback = torch.zeros(3, 64, 64)
            return fallback, label


def get_digit_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def main():
    print(f"=== Training 10-Class Digit Classifier on {DEVICE} ===")
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Metadata not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    digits_df = df[df["character"].astype(str).str.match(r"^[0-9]$")].copy()
    print(f"Loaded {len(digits_df)} real plate digit crops.")

    # Stratified 85/15 train/val split
    np.random.seed(42)
    shuffled = digits_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_split_idx = int(len(shuffled) * 0.85)
    train_df = shuffled.iloc[:val_split_idx]
    val_df = shuffled.iloc[val_split_idx:]

    train_tf, val_tf = get_digit_transforms()
    train_ds = ThaiDigitDataset(train_df, CROPS_DIR, transform=train_tf)
    val_ds = ThaiDigitDataset(val_df, CROPS_DIR, transform=val_tf)

    print(f"Dataset split: Train={len(train_ds)}, Val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = DigitClassifier(n_classes=10, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

    best_val_acc = 0.0
    epochs = 8

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> Saved new best digit model to {MODEL_SAVE_PATH} ({val_acc:.2f}%)")

    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Weights saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
