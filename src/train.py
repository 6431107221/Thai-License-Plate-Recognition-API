import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import json
import editdistance
from sklearn.metrics import f1_score

# Import จากไฟล์ที่เราสร้าง
from models import ResNetCRNN, ProvinceClassifier
from datasets import OCRDataset, ProvinceDataset, ocr_collate, get_ocr_transforms, get_prov_transforms
from utils import best_path_decode

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}")

CROPS_ROOT = Path("crops_all")
BATCH_SIZE_PROV = 32   
BATCH_SIZE_OCR = 32
EPOCHS = 50
EARLY_STOP = 15
NUM_WORKERS = 0 

TRAIN_UNIFIED = CROPS_ROOT / "train" / "train_unified.csv"
VAL_UNIFIED   = CROPS_ROOT / "valid" / "val_unified.csv"

# --- Helper: Filter Data ---
def filter_existing_provinces(df, root):
    valid_rows = []
    print("Filtering dataset (keeping only existing province crops)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_rel_plate = row["image"]
        img_rel_prov = img_rel_plate.replace("/plates/", "/provs/").replace("__plate", "__prov")
        if (root / img_rel_prov).exists():
            valid_rows.append(row)
    return pd.DataFrame(valid_rows)

# --- 1. Province Training ---
def train_province_model():
    print("\n--- Start Training Province Model ---")
    
    # 1. Load Data (CSV)
    if not TRAIN_UNIFIED.exists(): 
        print(f"Error: Unified CSV not found at {TRAIN_UNIFIED}. Run preprocess.py first.")
        return
    try:
        train_df_raw = pd.read_csv(TRAIN_UNIFIED).fillna("")
        val_df_raw = pd.read_csv(VAL_UNIFIED).fillna("")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return
    
    train_df = filter_existing_provinces(train_df_raw, CROPS_ROOT)
    val_df = filter_existing_provinces(val_df_raw, CROPS_ROOT)
    print(f"Train samples: {len(train_df_raw)} -> {len(train_df)}")

    # ============================================================
    #  Load Class Map จากโมเดลเก่า 
    # ============================================================
    forced_map = None
    LOAD_PATH = Path("ocr_minimal/province_best.pth") # โมเดลต้นฉบับ
    
    if LOAD_PATH.exists():
        print(f" Reading class map from {LOAD_PATH} to ensure consistency...")
        try:
            # FIX: เพิ่ม weights_only=False แก้ Warning
            ckpt = torch.load(LOAD_PATH, map_location=DEVICE, weights_only=False)
            if "class_map" in ckpt:
                saved_i2p = ckpt["class_map"]
                forced_map = {v: int(k) for k, v in saved_i2p.items()}
                print(f"  Loaded class map with {len(forced_map)} classes.")
        except Exception as e:
            print(f"  Failed to load class map: {e}")

    # 2. Create Dataset
    train_ds = ProvinceDataset(train_df, CROPS_ROOT, class_map=forced_map, training=True)
    val_ds = ProvinceDataset(val_df, CROPS_ROOT, class_map=train_ds.p2i, training=False)
    
    is_cuda = (DEVICE.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_PROV, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=is_cuda)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_PROV, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=is_cuda)

    # 3. Weights Setup
    master_map = train_ds.p2i
    all_labels = [master_map.get(row["gt_province"], 0) for _, row in train_df.iterrows()]
    class_counts = np.bincount(all_labels, minlength=len(master_map))
    class_counts = np.where(class_counts == 0, 1, class_counts)
    total_samples = len(all_labels)
    n_classes = len(master_map)
    
    class_weights = total_samples / (n_classes * class_counts)
    class_weights = np.clip(class_weights, 1.0, 10.0)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    print(f"Class Weights configured. Total Classes: {n_classes}")

    # 4. Model Setup
    model = ProvinceClassifier(len(train_ds.p2i)).to(DEVICE)
    best_f1 = 0.0
    patience_counter = 0

    # Load Weights
    if LOAD_PATH.exists():
        print(" Loading existing province weights...")
        try:
            # FIX: เพิ่ม weights_only=False แก้ Warning
            ckpt = torch.load(LOAD_PATH, map_location=DEVICE, weights_only=False)
            state_dict = ckpt["model_state"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith("model."): new_state_dict[f"model.{k}"] = v 
                else: new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            
            if "best_f1" in ckpt: best_f1 = ckpt["best_f1"]
            print(f"  Model weights loaded! Resuming with Best F1: {best_f1:.4f}")
        except Exception as e:
            print(f"  Load weights failed: {e}")

    # Optimizer & Loss
    # FIX: ลด LR เหลือ 1e-5 เพื่อ Fine-tuning ไม่ให้ weight เก่าพัง
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)

    # 4. Training Loop
    # Baseline Check
    model.eval()
    all_preds, all_labels_val = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=is_cuda):
                out = model(imgs)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels_val.extend(labels.numpy())
    current_val_f1 = f1_score(all_labels_val, all_preds, average='macro')
    print(f" Baseline Val F1: {current_val_f1:.4f}")

    if best_f1 == 0.0:
        best_f1 = current_val_f1
        print(f" initialized best_f1 to {best_f1:.4f}")

    for ep in range(EPOCHS):
        model.train()
        train_ds.training = True

        # ========================================================
        # 🔥 FIX: FREEZE BATCH NORM (แก้ปัญหา F1 ร่วงตรงนี้) 🔥
        # ========================================================
        # บังคับให้ Layer BatchNorm ใช้ค่า stat เดิม (ไม่คำนวณใหม่จากข้อมูลน้อยๆ)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        # ========================================================

        loss_sum = 0.0; correct = 0; total = 0

        pbar = tqdm(train_loader, desc=f"Ep {ep+1}/{EPOCHS} [Prov]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=is_cuda):
                out = model(imgs)
                loss = criterion(out, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.2%}")

        # Validation
        model.eval()
        val_ds.training = False
        all_preds, all_labels_val = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=is_cuda):
                    out = model(imgs)
                preds = out.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels_val.extend(labels.numpy())

        val_f1 = f1_score(all_labels_val, all_preds, average='macro')
        print(f"   └── Val F1: {val_f1:.4f} (Best: {best_f1:.4f})")

        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "class_map": train_ds.i2p,
                "best_f1": best_f1, 
                "epoch": ep
            }, "ocr_train_out/province_best.pth")
            print(f"       Model Saved! (New Best F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                # Save latest state anyway
                torch.save({"model_state": model.state_dict(), "class_map": train_ds.i2p}, "ocr_train_out/province_best.pth") 
                print("Early Stopping. Saving latest model state before exit.")
                break

# --- 2. OCR Training (Code เดิม) ---
def train_ocr_model():
    print("\n--- Start Training OCR Model ---")
    
    json_path = Path("ocr_minimal/int_to_char.json")
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        int_to_char = json.load(f)
    char_to_int = {v: int(k) for k, v in int_to_char.items()}

    if not TRAIN_UNIFIED.exists(): 
        print(f"Error: Unified CSV not found at {TRAIN_UNIFIED}. Run preprocess.py first.")
        return
    try:
        train_df = pd.read_csv(TRAIN_UNIFIED).fillna("")
        val_df = pd.read_csv(VAL_UNIFIED).fillna("")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return
    
    train_ds = OCRDataset(train_df, CROPS_ROOT, char_to_int, transform=get_ocr_transforms(True))
    val_ds = OCRDataset(val_df, CROPS_ROOT, char_to_int, transform=get_ocr_transforms(False))
    
    is_cuda = (DEVICE.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_OCR, shuffle=True, 
                              collate_fn=ocr_collate, num_workers=NUM_WORKERS, pin_memory=is_cuda)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_OCR, collate_fn=ocr_collate, 
                            num_workers=NUM_WORKERS, pin_memory=is_cuda)

    model = ResNetCRNN(1, len(int_to_char), hidden_size=256, num_rnn_layers=2).to(DEVICE)
    
    OCR_PRETRAINED_PATH = Path("ocr_minimal/best_model.pth")
    OCR_SAVE_PATH = Path("ocr_train_out/best_model.pth")
    
    if OCR_PRETRAINED_PATH.exists():
        print(f" Loading existing OCR model from {OCR_PRETRAINED_PATH}...")
        try:
            ckpt = torch.load(OCR_PRETRAINED_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            print(" Model loaded successfully!")
        except Exception as e:
            print(f" Failed to load model: {e}. Training from scratch.")

    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)

    best_val_cer = 1.0
    
    print("Checking baseline performance...")
    model.eval()
    cer_sum = 0; tot = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs, tg, tg_lens, _, texts, names = batch
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            preds = best_path_decode(out, int_to_char)
            idx = 0
            for i, L in enumerate(tg_lens):
                gt = "".join(int_to_char[str(x)] for x in tg[idx:idx+int(L)].tolist())
                div = max(1, len(gt))
                cer_sum += editdistance.eval(preds[i], gt) / div
                tot += 1; idx += int(L)
    val_cer = cer_sum / max(1, tot)
    print(f"Baseline CER: {val_cer:.4f}")
    if val_cer < 1.0: best_val_cer = val_cer

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"OCR Ep {epoch}")
        
        for batch in pbar:
            imgs, tg, tg_lens, _, texts, names = batch
            imgs, tg, tg_lens = imgs.to(DEVICE), tg.to(DEVICE), tg_lens.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=is_cuda):
                out = model(imgs) 
                logp = out.log_softmax(-1)
                logp_loss = logp.permute(1, 0, 2)
                
                input_lengths = torch.full((imgs.size(0),), out.size(1), dtype=torch.long).to(DEVICE)
                loss = criterion(logp_loss, tg, input_lengths, tg_lens)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) 
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        model.eval()
        cer_sum = 0; tot = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, tg, tg_lens, _, texts, names = batch
                imgs = imgs.to(DEVICE)
                out = model(imgs)
                preds = best_path_decode(out, int_to_char)
                idx = 0
                for i, L in enumerate(tg_lens):
                    gt = "".join(int_to_char[str(x)] for x in tg[idx:idx+int(L)].tolist())
                    div = max(1, len(gt))
                    cer_sum += editdistance.eval(preds[i], gt) / div
                    tot += 1; idx += int(L)
        
        val_cer = cer_sum / max(1, tot)
        avg_loss = total_loss / len(train_loader)
        print(f"[E{epoch}] Loss={avg_loss:.4f} | CER={val_cer:.4f}")

        scheduler.step(val_cer)

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            Path("ocr_train_out").mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "int_to_char": int_to_char,
                "epoch": epoch,
                "cer": val_cer
            }, OCR_SAVE_PATH)
            print(f">> Saved New Best Model (CER: {val_cer:.4f})")

if __name__ == "__main__":
    train_province_model()
    train_ocr_model()