"""
src/train_lao_province.py

Trains MobileNetV2 for 18 Lao Provinces on the 11,317 Lao plate crops.
Features:
  - Class-weighted Cross-Entropy Loss to balance thin classes against Vientiane Capital
  - Evaluates Top-1 and Top-3 accuracy on the official validation split
  - Saves best checkpoint to weights/province_model_lao.pth
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAO_DIR = PROJECT_ROOT / "datasets" / "lao-plate-dataset"
TRAIN_CSV = LAO_DIR / "ground_truth_train.csv"
VAL_CSV = LAO_DIR / "ground_truth_validation.csv"
IMAGES_DIR = LAO_DIR / "images"

WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = WEIGHTS_DIR / "province_model_lao.pth"
MAP_PATH = WEIGHTS_DIR / "province_map_lao.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

class LaoProvinceDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Filter to existing images
        valid_rows = []
        for _, row in self.df.iterrows():
            p = self.images_dir / row["filename"]
            if p.exists() and p.stat().st_size > 500:
                valid_rows.append(row)
        self.df = pd.DataFrame(valid_rows)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row["filename"]
        img = Image.open(img_path).convert("RGB")
        label = int(row["province_id"])

        if self.transform:
            img = self.transform(img)

        return img, label

def compute_class_weights(df, n_classes=18):
    counts = np.zeros(n_classes, dtype=np.float32)
    for c in df["province_id"].values:
        counts[int(c)] += 1
    total = float(len(df))
    # Inverse frequency smoothing
    weights = total / (n_classes * (counts + 1e-5))
    weights = np.clip(weights, 0.2, 20.0)  # Clip extremes
    return torch.tensor(weights, dtype=torch.float32)

def train_lao_province_model(epochs=15, batch_size=32, lr=2e-4):
    print(f"\n--- Training Lao Province Classifier on device: {DEVICE} ---")
    
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        province_map = json.load(f)
    n_classes = len(province_map)
    print(f"Number of Lao provinces: {n_classes}")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=6, translate=(0.04, 0.04)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = LaoProvinceDataset(TRAIN_CSV, IMAGES_DIR, train_tf)
    val_ds = LaoProvinceDataset(VAL_CSV, IMAGES_DIR, val_tf)

    print(f"Loaded {len(train_ds)} train samples and {len(val_ds)} validation samples.")

    class_weights = compute_class_weights(train_ds.df, n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Backbone: MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, n_classes)
    )
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_top1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_acc = correct / total
        train_loss = train_loss / total

        # Validation
        model.eval()
        val_correct_1 = 0
        val_correct_3 = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                
                # Top-1
                preds1 = outputs.argmax(dim=1)
                val_correct_1 += (preds1 == labels).sum().item()
                
                # Top-3
                _, top3 = outputs.topk(3, dim=1)
                val_correct_3 += sum([label.item() in top3[i] for i, label in enumerate(labels)])
                val_total += len(labels)

        val_acc1 = val_correct_1 / val_total
        val_acc3 = val_correct_3 / val_total

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Top-1: {val_acc1*100:.2f}% | Val Top-3: {val_acc3*100:.2f}%")

        if val_acc1 > best_top1:
            best_top1 = val_acc1
            torch.save({
                "model_state": model.state_dict(),
                "val_acc_top1": val_acc1,
                "val_acc_top3": val_acc3,
                "province_map": province_map,
            }, str(SAVE_PATH))

    print(f"\nTraining Complete! Best Validation Top-1: {best_top1*100:.2f}%")
    print(f"Saved weights to: {SAVE_PATH}")

if __name__ == "__main__":
    train_lao_province_model()
