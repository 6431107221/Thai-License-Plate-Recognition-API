
import torch 
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pandas as pd
import numpy as np
import random
from pathlib import Path
from preprocess import get_prov_transforms

class OCRDataset(Dataset):
    def __init__(self, df, root, char_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.char_map = char_map
        self.transform = transform

    def encode(self, text):
        encoded = [self.char_map.get(char, 0) for char in text]
        return torch.LongTensor(encoded)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["image"]
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new('RGB', (256, 64))

        if self.transform:
            img = self.transform(img)

        target = self.encode(row["gt_plate"])
        return img, target, len(target), row["gt_plate"], str(img_path), "real"

    def __len__(self): return len(self.df)

class ProvinceDataset(Dataset):
    def __init__(self, df, root, class_map=None, training=True):
        self.df = df.reset_index(drop=True).fillna("")
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
        img_rel_prov = img_rel_plate.replace("/plates/", "/provs/").replace("_plate", "_prov")
        img_path = self.root / img_rel_prov
        
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new('RGB', (224, 224))
            
        if self.transform:
            img = self.transform(img)
        prov_class_id = self.p2i.get(row["gt_province"], 0)
        return img, prov_class_id
    
    def __len__(self): return len(self.df)

def ocr_collate(batch):
    imgs, targets, target_lens, texts, paths, types = zip(*batch)
    imgs = torch.stack(imgs)
    targets = torch.cat(targets)
    target_lens = torch.LongTensor(target_lens)
    return imgs, targets, target_lens, texts, paths, types

# --- Synthetic Datasets ---

class SyntheticOCRDataset(Dataset):
    def __init__(self, size=1500, char_to_int=None, transform=None): # Reduced size for balance
        self.size = size
        self.char_map = char_to_int
        self.transform = transform
        
        # Priority Search for Thai Fonts (High quality/Real-look first)
        maybe_fonts = [
            "C:/Users/Simai/AppData/Local/Microsoft/Windows/Fonts/THSarabunNew Bold.ttf",
            "C:/Windows/Fonts/tahoma.ttf", 
        ]
        self.font_path = "C:/Windows/Fonts/tahoma.ttf"
        for p in maybe_fonts:
            if Path(p).exists():
                self.font_path = p
                print(f" OCR Synthetic using font: {self.font_path}")
                break
        
        self.thai_chars = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
        self.nums = "0123456789"

    def _gen_text(self):
        if random.random() > 0.5:
            c = "".join(random.choices(self.thai_chars, k=2))
            n = str(random.randint(1, 9999))
            return f"{c} {n}"
        else:
            c = "".join(random.choices(self.thai_chars, k=2))
            digit = str(random.randint(1, 9))
            n = str(random.randint(1, 9999))
            return f"{digit}{c} {n}"

    def __getitem__(self, idx):
        W, H = 256, 64
        Bg_type = random.choices(
            [(255, 255, 255), (255, 240, 150), (200, 255, 200), (255, 200, 200), (200, 240, 255)],
            weights=[0.6, 0.15, 0.05, 0.1, 0.1], k=1
        )[0]
        bg_color = (
            max(0, min(255, Bg_type[0] + random.randint(-15, 15))),
            max(0, min(255, Bg_type[1] + random.randint(-15, 15))),
            max(0, min(255, Bg_type[2] + random.randint(-15, 15)))
        )
        img = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(img)
        text = self._gen_text()
        
        try:
            # Increased font size for better clarity
            font = ImageFont.truetype(self.font_path, 55)
        except:
            font = ImageFont.load_default()
            
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x = (W - tw) // 2 + random.randint(-10, 10)
        y = (H - th) // 2 + random.randint(-5, 5)
        draw.text((x, y), text, font=font, fill=(0, 0, 0))
        
        # PERSPECTIVE SKEW (Slightly reduced for stability)
        if random.random() > 0.7:
            skew_x = random.uniform(-0.1, 0.1)
            skew_y = random.uniform(-0.03, 0.03)
            img = img.transform((W, H), Image.AFFINE, (1, skew_x, 0, skew_y, 1, 0), resample=Image.BICUBIC, fillcolor=bg_color)
            
        if random.random() > 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))
            
        if self.transform:
            img = self.transform(img) 
            
        target = torch.LongTensor([self.char_map.get(c, 0) for c in text if c in self.char_map])
        return img, target, len(target), text, "synthetic", "syn"

    def __len__(self): return self.size

class SyntheticProvinceDataset(Dataset):
    def __init__(self, master_dataset, size=2000, transform=None):
        self.size = size
        self.p2i = master_dataset.p2i
        self.provinces = list(self.p2i.keys())
        self.transform = transform
        
        maybe_fonts = [
            "C:/Users/Simai/AppData/Local/Microsoft/Windows/Fonts/THSarabunNew Bold.ttf",
            "C:/Windows/Fonts/tahoma.ttf"
        ]
        self.font_path = "C:/Windows/Fonts/tahoma.ttf"
        for p in maybe_fonts:
            if Path(p).exists():
                self.font_path = p
                break

    def __getitem__(self, idx):
        W, H = 200, 60
        Bg_type = random.choices(
            [(255, 255, 255), (255, 240, 150), (255, 200, 200)],
            weights=[0.6, 0.3, 0.1], k=1
        )[0]
        bg_color = (
            max(0, min(255, Bg_type[0] + random.randint(-15, 15))),
            max(0, min(255, Bg_type[1] + random.randint(-15, 15))),
            max(0, min(255, Bg_type[2] + random.randint(-15, 15)))
        )
        img = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(img)
        prov_name = random.choice(self.provinces)
        
        try:
            font = ImageFont.truetype(self.font_path, 35)
        except:
            font = ImageFont.load_default()
            
        bbox = draw.textbbox((0, 0), prov_name, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x, y = (W - tw) // 2 + random.randint(-5, 5), (H - th) // 2 + random.randint(-3, 3)
        draw.text((x, y), prov_name, font=font, fill=(0, 0, 0))
        
        if random.random() > 0.4:
            img = img.filter(ImageFilter.GaussianBlur(1))
            
        if self.transform:
            img = self.transform(img)
        return img, self.p2i.get(prov_name, 0)
        
    def __len__(self): return self.size
