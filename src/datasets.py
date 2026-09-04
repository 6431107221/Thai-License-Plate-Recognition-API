
import torch 
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pandas as pd
import numpy as np
import random
from pathlib import Path
try:
    from src.validators import is_valid_plate
except ImportError:
    # Fallback if running from a different context where src is not in path
    import sys
    sys.path.append(str(Path(__file__).parent))
    from validators import is_valid_plate


class OCRDataset(Dataset):
    def __init__(self, df, root, char_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.char_map = char_map
        self.int_to_char = {i: c for c, i in self.char_map.items()}
        self.transform = transform

    def encode(self, text):
        # Filter to valid characters and ensure non-blank tokens for CTC
        encoded = [self.char_map[char] for char in str(text) if char in self.char_map and self.char_map[char] != 0]
        return torch.LongTensor(encoded)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row.get("plate_image") or row.get("image")
        img_path = self.root / str(rel_path)
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new('RGB', (256, 64))

        if self.transform:
            img = self.transform(img)

        target = self.encode(row["gt_plate"])
        return img, target, len(target), str(row["gt_plate"]), str(img_path), "real"

    def __len__(self): return len(self.df)

class ProvinceDataset(Dataset):
    def __init__(self, df, root, char_map, transform=None):
        self.df = df.reset_index(drop=True).fillna("")
        self.root = Path(root)
        self.transform = transform
        self.char_map = char_map
        self.int_to_char = {i: p for p, i in self.char_map.items()}

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row.get("province_image") or row.get("image")
        img_path = self.root / str(rel_path)
        
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new('RGB', (224, 224))
            
        if self.transform:
            img = self.transform(img)
        prov_class_id = self.char_map.get(str(row["gt_province"]).strip(), 0)
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
            "C:/Users/Simai/AppData/Local/Microsoft/Windows/Fonts/THSarabunNew Bold.ttf" 
        ]
        for p in maybe_fonts:
            if Path(p).exists():
                self.font_path = p
                print(f" OCR font: {self.font_path}")
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
        RGB_Color = [(255, 255, 255), (255, 240, 150), (200, 255, 200), (255, 200, 200), (200, 240, 255)]
        Bg_type = random.choices(RGB_Color,weights=[0.6, 0.15, 0.05, 0.1, 0.1], k=1)[0]
        bg_color = (
            max(0, min(255, Bg_type[0] + random.randint(-15, 15))),
            max(0, min(255, Bg_type[1] + random.randint(-15, 15))),
            max(0, min(255, Bg_type[2] + random.randint(-15, 15)))
        )
        img = Image.new("RGB", (W, H), bg_color)

        text = self._gen_text()
        try:
            font = ImageFont.truetype(self.font_path, 55)
        except:
            font = ImageFont.load_default()
        
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x = (W - tw) // 2 + random.randint(-10, 10)
        y = (H - th) // 2 + random.randint(-5, 5)
        draw.text((x, y), text, font=font, fill=(0, 0, 0))
        
        if random.random() > 0.7:
            skew_x = random.uniform(-0.1, 0.1)
            skew_y = random.uniform(-0.03, 0.03)
                                                      #ขยายกว้าง,บิดแนวนอน,เลื่อนซ้ายขวา,บิดแนวตั้ง,ขยายสูง,เลื่อนขึ้นลง
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
        self.char_map = master_dataset.char_map
        self.provinces = list(self.char_map.keys())
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
        prov_name = random.choice(self.provinces)
        
        try:
            font = ImageFont.truetype(self.font_path, 35)
        except:
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), prov_name, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x, y = (W - tw) // 2 + random.randint(-5, 5), (H - th) // 2 + random.randint(-3, 3)
        draw.text((x, y), prov_name, font=font, fill=(0, 0, 0))
        
        if random.random() > 0.4:
            img = img.filter(ImageFilter.GaussianBlur(1))
            
        if self.transform:
            img = self.transform(img)
        return img, self.char_map.get(prov_name, 0)
        
    def __len__(self): return self.size
