"""
src/train_country_classifier.py

Trains a lightweight Country Classifier (Model 1.5: Thai vs Laos) on rectified license plate crops:
- Class 0: Thai
- Class 1: Laos
Uses MobileNetV3-Small for sub-2ms inference with near-100% accuracy.
Saves weights to weights/country_classifier.pth.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = WEIGHTS_DIR / "country_classifier.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

class CountryDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def get_country_data():
    # 1. Collect Thai crops
    thai_dirs = [
        PROJECT_ROOT / "output" / "train" / "rectified_plates",
        PROJECT_ROOT / "output" / "ground_truth_crops" / "rectified_plates",
    ]
    thai_files = []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for d in thai_dirs:
        if d.exists():
            thai_files.extend([p for p in d.glob("*") if p.suffix.lower() in exts and p.stat().st_size > 500])

    # 2. Collect Diverse Lao crops (Roboflow in-the-wild vehicle crops + toll distinct images)
    lao_files = []
    lao_wild_dir = PROJECT_ROOT / "datasets" / "Lao" / "lao_plate_crops"
    if lao_wild_dir.exists():
        lao_files.extend([p for p in lao_wild_dir.glob("*.jpg") if p.stat().st_size > 300])

    lao_distinct_dir = PROJECT_ROOT / "datasets" / "Lao" / "lao-plate-dataset" / "distinct_images"
    if lao_distinct_dir.exists():
        lao_distinct_all = [p for p in lao_distinct_dir.glob("*.jpg") if p.stat().st_size > 500]
        random.seed(42)
        lao_files.extend(random.sample(lao_distinct_all, min(1200, len(lao_distinct_all))))

    print(f"Discovered {len(thai_files)} Thai crops and {len(lao_files)} diverse Lao crops.")

    # Balance datasets
    n_samples = min(len(thai_files), len(lao_files), 3000)
    random.seed(42)
    thai_sampled = random.sample(thai_files, n_samples)
    lao_sampled = random.sample(lao_files, n_samples)

    labeled_data = [(p, 0) for p in thai_sampled] + [(p, 1) for p in lao_sampled]
    random.shuffle(labeled_data)

    # 85% train, 15% val
    split_idx = int(len(labeled_data) * 0.85)
    train_samples = labeled_data[:split_idx]
    val_samples = labeled_data[split_idx:]

    print(f"Dataset balanced: {len(train_samples)} train, {len(val_samples)} validation.")
    return train_samples, val_samples

def train_country_classifier(epochs=10, batch_size=32, lr=1e-3):
    print(f"\n--- Training Country Classifier on device: {DEVICE} ---")
    train_samples, val_samples = get_country_data()

    train_tf = transforms.Compose([
        transforms.Resize((128, 256)),
        transforms.RandomAffine(degrees=5, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((128, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(CountryDataset(train_samples, train_tf), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CountryDataset(val_samples, val_tf), batch_size=batch_size, shuffle=False)

    # Lightweight MobileNetV3-Small backbone
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)  # 0: Thai, 1: Lao
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0

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
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "classes": {0: "Thai", 1: "Laos"},
            }, str(SAVE_PATH))

    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Saved weights to: {SAVE_PATH}")

if __name__ == "__main__":
    train_country_classifier()
