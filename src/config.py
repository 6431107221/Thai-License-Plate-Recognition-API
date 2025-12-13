import torch
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 1. System & Paths
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0 
    
    # Paths ข้อมูล
    SOURCE_DATASET_DIR = PROJECT_ROOT / "OCR-Car-Plate_dataset"
    CROPS_DIR = PROJECT_ROOT / "crops_all"
    
    # CSV Files
    TRAIN_CSV = CROPS_DIR / "train" / "train_unified.csv"
    VAL_CSV   = CROPS_DIR / "valid" / "val_unified.csv"
    TEST_CSV  = CROPS_DIR / "test" / "test_unified.csv"

    # ==========================================
    # Model Paths
    # 1. Pretrained Models
    PRETRAINED_DIR = PROJECT_ROOT / "ocr_minimal"
    OCR_PRETRAINED_PATH  = PRETRAINED_DIR / "best_model.pth"
    PROV_PRETRAINED_PATH = PRETRAINED_DIR / "province_best.pth"
    
    # 2. Output Trained Models
    OUTPUT_DIR = PROJECT_ROOT / "ocr_train_out"
    OCR_MODEL_SAVE_PATH  = OUTPUT_DIR / "best_model.pth"
    PROV_MODEL_SAVE_PATH = OUTPUT_DIR / "province_best.pth"
    
    # Mapping Files
    CHAR_MAP_PATH = PRETRAINED_DIR / "int_to_char.json"
    
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # ==========================================
    # 2. Roboflow(API)
    RF_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    MODEL_DETECTION_ID = "car-plate-detection-ahcak/3" 
    MODEL_OCR_PREP_ID  = "ocr_prepare_test-tfc9g/4"    

    # ==========================================
    # 3. Training Hyperparameters
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 15
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 5e-4
    BATCH_SIZE_OCR = 32
    BATCH_SIZE_PROV = 32

    # ==========================================
    # 4. Image Augmentation 
    OCR_TARGET_SIZE = (64, 256)
    PROV_TARGET_SIZE = (224, 224)
    
    AUG_DEGREES = 15
    AUG_TRANSLATE = (0.05, 0.05) 
    AUG_SCALE = (0.8, 1.2)
    AUG_SHEAR = 10 
    AUG_PERSPECTIVE = 0.5 
    
    AUG_COLOR_JITTER = (0.5, 0.5, 0.5, 0.1)
    AUG_BLUR_SIGMA = (0.1, 1.5)

    # ==========================================
    # 5. Debug
    DEBUG_MODE = True
    DEBUG_IMAGE_DIR = PROJECT_ROOT / "debug_images"

cfg = Config()

cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cfg.DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)