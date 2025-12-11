import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from pathlib import Path
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np

class SmartResize:
    def __init__(self, target_size, fill=0, is_ocr=False):
        """
        target_size: (height, width) for OCR, (height, width) for Province
        is_ocr: ถ้าเป็น OCR เราจะ fix height แล้วปล่อย width
        """
        self.target_size = target_size 
        self.fill = fill
        self.is_ocr = is_ocr

    def __call__(self, img):
        tgt_h, tgt_w = self.target_size
        w, h = img.size

        if self.is_ocr:
            # OCR: ยึดความสูงเป็นหลัก, ความกว้างปรับตาม
            scale = tgt_h / h
            new_h = tgt_h
            new_w = int(w * scale)
            # ถ้า new_w เกิน tgt_w ให้ยึดความกว้างแทน
            if new_w > tgt_w:
                scale = tgt_w / w
                new_w = tgt_w
                new_h = int(h * scale)
        else:
            # Province: ยึดด้านที่ยาวสุด
            scale = min(tgt_h / h, tgt_w / w)
            new_h = int(h * scale)
            new_w = int(w * scale)

        img = img.resize((new_w, new_h), resample=Image.BICUBIC)

        # 3. Create Background & Paste (Padding)
        # สร้างภาพพื้นหลังสีดำ (หรือสีเทาค่า 0)
        # ถ้าภาพเดิมเป็น L (Gray) พื้นหลังก็ L, ถ้า RGB พื้นหลังก็ RGB
        new_img = Image.new(img.mode, (tgt_w, tgt_h), self.fill)
        
        # คำนวณตำแหน่งวางตรงกลาง
        paste_x = (tgt_w - new_w) // 2
        paste_y = (tgt_h - new_h) // 2
        
        new_img.paste(img, (paste_x, paste_y))
        return new_img

# --- Transforms Config ---

def get_ocr_transforms(is_train=True):
    base_transforms = [
        SmartResize((64, 256), is_ocr=True),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ]
    
    if is_train:
        augments = [
            # 1. บิดมุม
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.8, 1.2), shear=10, fill=0),
            T.RandomPerspective(distortion_scale=0.5, p=0.4),

            # 2.เพิ่ม GaussianBlur
            T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))], p=0.3),
            
            # 3. เพิ่ม Noise
            T.Lambda(lambda x: x + 0.05 * torch.randn_like(x) if torch.is_tensor(x) else x),
            
            # 4. Random Invert
            T.RandomInvert(p=0.1),
        ]
        return T.Compose(augments + base_transforms)
    else:
        return T.Compose(base_transforms)

# Province 
def get_prov_transforms(is_train=True):
    ops = []
    ops.append(SmartResize((224, 224), is_ocr=False)) # Resize ก่อน
    
    if is_train:
        ops.append(T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.1), shear=10))
        ops.append(T.RandomPerspective(distortion_scale=0.2, p=0.3))
        ops.append(T.ColorJitter(brightness=0.5, contrast=0.5)) 
        
    ops.append(T.ToTensor())
    
    ops.append(T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))
    
    ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(ops)

# --- Datasets ---
class OCRDataset(Dataset):
    def __init__(self, df, root, char_to_int, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.cti = char_to_int
        self.transform = transform

    def encode(self, txt):
        txt = str(txt) if txt is not None else ""
        return torch.tensor([self.cti[c] for c in txt if c in self.cti], dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["image"]
        try:
            img = Image.open(img_path).convert("L")
        except:
            img = Image.new('L', (256, 64))

        if self.transform:
            img = self.transform(img)

        target = self.encode(row["gt_plate"])
        return img, target, len(target), row["gt_plate"], str(img_path)

    def __len__(self): return len(self.df)

class ProvinceDataset(Dataset):
    def __init__(self, df, root, class_map=None, training=True):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.training = training
        
        if class_map is not None:
            self.p2i = class_map
            self.i2p = {i:p for p,i in self.p2i.items()}
        else:
            self.provs = sorted(df["gt_province"].unique())
            self.p2i = {p:i for i,p in enumerate(self.provs)}
            self.i2p = {i:p for p,i in self.p2i.items()}

        self.transform = get_prov_transforms(training)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_plate = row["image"]
  
        img_rel_prov = img_rel_plate.replace("/plates/", "/provs/").replace("__plate", "__prov")
        img_path = self.root / img_rel_prov

        try:
  
            img = Image.open(img_path).convert("L") 
        except:
            img = Image.new("L", (224, 224))

        img = self.transform(img)
        label = self.p2i.get(row["gt_province"], 0)
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self): return len(self.df)

def ocr_collate(batch):
    imgs, tg, lens, texts, names = zip(*batch)
    return torch.stack(imgs), torch.cat(tg), torch.tensor(lens, dtype=torch.long), None, texts, names