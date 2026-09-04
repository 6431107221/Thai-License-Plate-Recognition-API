import os
import re
import shutil
import zipfile
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAO_DATASET_DIR = PROJECT_ROOT / "datasets" / "lao-plate-dataset"
SRC_IMG_DIR = LAO_DATASET_DIR / "images"
GT_ALL_CSV = LAO_DATASET_DIR / "ground_truth_all.csv"

DISTINCT_DIR = LAO_DATASET_DIR / "distinct_images"
DISTINCT_CSV = LAO_DATASET_DIR / "ground_truth_distinct.csv"
ZIP_PATH = LAO_DATASET_DIR / "lao_plate_distinct_dataset.zip"

# Strict Lao plate pattern: 1 to 2 Lao consonants (\u0E81-\u0EAE) followed by 1 to 5 digits (0-9)
LAO_PLATE_REGEX = re.compile(r"^[\u0E81-\u0EAE]{1,2}\s*\d{1,5}$")


def is_valid_lao_plate(text: str) -> bool:
    """Verifies that the plate text contains valid Lao consonants and Arabic numerals only."""
    if not isinstance(text, str):
        return False
    return bool(LAO_PLATE_REGEX.match(text.strip()))


def compute_image_quality(img_path: Path) -> float:
    """Computes Laplacian variance (sharpness) and resolution score."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        h, w = gray.shape
        # Sharpness score weighted by resolution
        score = float(lap_var) * (1.0 + np.log10(max(h * w, 1000) / 1000.0))
        return score
    except Exception:
        return 0.0


def extract_distinct_dataset():
    print(f"[1/6] Loading Ground Truth: {GT_ALL_CSV}")
    df = pd.read_csv(GT_ALL_CSV)
    total_records = len(df)
    print(f"      Total raw records loaded: {total_records}")

    # Step 2: Clean & filter out invalid/corrupted records (Thai characters, pure numbers, Latin text)
    print("[2/6] Filtering out noise, pure numbers, and non-Lao characters...")
    valid_mask = df["plate_text"].apply(is_valid_lao_plate)
    dropped_count = (~valid_mask).sum()
    df_clean = df[valid_mask].copy()
    print(f"      Filtered out {dropped_count} corrupted/noisy rows.")
    print(f"      Clean Lao records remaining: {len(df_clean)}")

    # Step 3: Group by distinct (plate_text, province) to preserve unique vehicle registrations across provinces
    print("[3/6] Grouping by distinct (plate_text, province)...")
    grouped = df_clean.groupby(["plate_text", "province"], as_index=False)
    distinct_count = len(grouped)
    print(f"      Identified {distinct_count} distinct (plate_text, province) combinations.")

    DISTINCT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean existing distinct images directory
    for existing_file in DISTINCT_DIR.glob("*.jpg"):
        existing_file.unlink()

    selected_rows = []

    print("[4/6] Selecting highest-quality image per distinct plate (sharpness & clarity)...")
    for (p_text, prov), group in tqdm(grouped, desc="Deduplicating"):
        best_row = None
        best_score = -1.0

        if len(group) == 1:
            best_row = group.iloc[0].to_dict()
            fn = best_row["filename"]
            img_path = SRC_IMG_DIR / fn
            best_row["sharpness_score"] = round(compute_image_quality(img_path), 2)
            best_row["duplicate_count"] = 1
        else:
            # Score each duplicate candidate and select the sharpest
            for _, row in group.iterrows():
                fn = row["filename"]
                img_path = SRC_IMG_DIR / fn
                if not img_path.exists():
                    continue
                score = compute_image_quality(img_path)
                if score > best_score:
                    best_score = score
                    best_row = row.to_dict()

            if best_row is not None:
                best_row["sharpness_score"] = round(best_score, 2)
                best_row["duplicate_count"] = len(group)
            else:
                best_row = group.iloc[0].to_dict()
                best_row["sharpness_score"] = 0.0
                best_row["duplicate_count"] = len(group)

        # Copy distinct image
        src_file = SRC_IMG_DIR / best_row["filename"]
        dst_file = DISTINCT_DIR / best_row["filename"]
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            best_row["image_path"] = f"images/{best_row['filename']}"
            selected_rows.append(best_row)

    # Step 5: Export clean matching ground truth CSV
    print(f"[5/6] Writing matching distinct ground truth CSV: {DISTINCT_CSV}")
    df_distinct = pd.DataFrame(selected_rows)

    col_order = [
        "filename",
        "image_path",
        "plate_text",
        "province",
        "letters",
        "digits",
        "province_id",
        "color",
        "sharpness_score",
        "duplicate_count",
        "split",
        "source",
    ]
    cols = [c for c in col_order if c in df_distinct.columns] + [
        c for c in df_distinct.columns if c not in col_order
    ]
    df_distinct = df_distinct[cols]
    # Sort cleanly by province and plate_text
    df_distinct = df_distinct.sort_values(by=["province", "plate_text"]).reset_index(drop=True)
    df_distinct.to_csv(DISTINCT_CSV, index=False, encoding="utf-8")
    print(f"      Saved {len(df_distinct)} rows to {DISTINCT_CSV}")

    # Step 6: Package into compressed zip
    print(f"[6/6] Compressing distinct images and matching CSV to: {ZIP_PATH}")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(DISTINCT_CSV, arcname="ground_truth_distinct.csv")
        for img_file in tqdm(sorted(DISTINCT_DIR.glob("*.jpg")), desc="Zipping images"):
            zf.write(img_file, arcname=f"images/{img_file.name}")

    zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSUCCESS! Clean distinct Lao dataset package ready:")
    print(f"  - Distinct Images: {len(df_distinct)} files in {DISTINCT_DIR}")
    print(f"  - Clean Matching Ground Truth: {DISTINCT_CSV}")
    print(f"  - Compressed Archive: {ZIP_PATH} ({zip_size_mb:.2f} MB)")


if __name__ == "__main__":
    extract_distinct_dataset()
