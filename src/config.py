import torch
from pathlib import Path

class Config:
    # --- System & Paths ---
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0 
    
    # Data Paths
    CROPS_DIR = PROJECT_ROOT / "crops_all"
    TRAIN_CSV = CROPS_DIR / "train" / "train_unified.csv"
    VAL_CSV   = CROPS_DIR / "valid" / "val_unified.csv"
    TEST_CSV  = CROPS_DIR / "test" / "test_unified.csv"

    # --- Model Weights ---
    WEIGHTS_DIR = PROJECT_ROOT / "weights"
    OCR_MODEL_SAVE_PATH  = WEIGHTS_DIR / "ocr_model.pth"
    PROV_MODEL_SAVE_PATH = WEIGHTS_DIR / "province_model.pth"
    
    # YOLO Models
    MODEL_DETECTION_PATH = WEIGHTS_DIR / "plate_detector.pt" 
    MODEL_OCR_PREP_PATH  = WEIGHTS_DIR / "component_detector.pt"    

    # Misc
    CHAR_MAP_PATH = WEIGHTS_DIR / "int_to_char.json"
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # --- Hyperparameters ---
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 20
    LEARNING_RATE = 1e-6
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE_OCR = 32
    BATCH_SIZE_PROV = 32

    # --- Transformations ---
    OCR_TARGET_SIZE = (64, 256)
    PROV_TARGET_SIZE = (224, 224)
    
    AUG_DEGREES = 15
    AUG_TRANSLATE = (0.05, 0.05) 
    AUG_SCALE = (0.8, 1.2)
    AUG_SHEAR = 10 
    AUG_PERSPECTIVE = 0.5 
    AUG_COLOR_JITTER = (0.5, 0.5, 0.5, 0.1)
    AUG_BLUR_SIGMA = (0.1, 1.5)

    # --- Debugging ---
    DEBUG_MODE = True
    DEBUG_IMAGE_DIR = PROJECT_ROOT / "debug_api"

cfg = Config()
cfg.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
cfg.DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)