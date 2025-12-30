import io
import numpy as np
from PIL import Image
from torchvision import transforms

class SmartResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        #pillow image
        if hasattr(img, 'size'):
            w, h = img.size
        #numpy array
        else:
            h, w = img.shape[:2]
            img = Image.fromarray(img)

        target_w, target_h = self.size
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        img_resized = img.resize((new_w, new_h), Image.BICUBIC) #Nearest Neighbor /Bilinear
        
        new_img = Image.new("L", (target_w, target_h), 0)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_img.paste(img_resized, (paste_x, paste_y))
        
        return new_img

def preprocess_raw_api_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_ocr_transforms(is_train=False):
    if is_train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            SmartResize((128, 64)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5)
            ], p=0.4),
            transforms.RandomAffine(degrees=3, translate=(0.02, 0.05)),
            transforms.ToTensor(), # 0-1
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            SmartResize((128, 64)),
            transforms.ToTensor(),
        ])

def get_prov_transforms(is_train=False):
    if is_train:
        return transforms.Compose([
            SmartResize((224, 224)),
            transforms.RandomAffine(degrees=5, translate=(0.02, 0.05)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3)
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            SmartResize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])