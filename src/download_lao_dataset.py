"""
src/download_lao_dataset.py

Downloads the Lao License Plate dataset from Hugging Face (goftaidc/lao-plate-dataset):
- Downloads parquet files for train, validation, and test splits
- Exports clean ground truth CSV files (train, val, test, combined)
- Extracts the actual JPEG images from the parquet files into datasets/lao-plate-dataset/images/
"""

import urllib.request
from pathlib import Path
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAO_DATASET_DIR = PROJECT_ROOT / "datasets" / "lao-plate-dataset"
DATA_DIR = LAO_DATASET_DIR / "data"
IMAGES_DIR = LAO_DATASET_DIR / "images"

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": "https://huggingface.co/datasets/goftaidc/lao-plate-dataset/resolve/main/data/train-00000-of-00001.parquet",
    "validation": "https://huggingface.co/datasets/goftaidc/lao-plate-dataset/resolve/main/data/validation-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/goftaidc/lao-plate-dataset/resolve/main/data/test-00000-of-00001.parquet",
}

def download_and_extract_lao_dataset():
    dfs = []

    for split_name, url in SPLITS.items():
        parquet_path = DATA_DIR / f"{split_name}.parquet"
        print(f"\n--- Processing Split: {split_name} ---")

        # 1. Download parquet file if not exists or if empty/pointer
        if not parquet_path.exists() or parquet_path.stat().st_size < 10000:
            print(f"Downloading {split_name} parquet from Hugging Face...")
            urllib.request.urlretrieve(url, str(parquet_path))
            print(f"Downloaded {split_name}.parquet ({parquet_path.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"Found existing {parquet_path.name} ({parquet_path.stat().st_size / (1024*1024):.2f} MB)")

        # 2. Read parquet
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} rows for {split_name}")

        # 3. Extract images and build ground truth table
        records = []
        extract_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split_name} images"):
            fname = row["filename"]
            img_dest = IMAGES_DIR / fname

            # Extract image bytes if file is a pointer (<1000 bytes) or doesn't exist
            if not img_dest.exists() or img_dest.stat().st_size < 1000:
                img_data = row["image"]
                if isinstance(img_data, dict) and "bytes" in img_data:
                    with open(img_dest, "wb") as f:
                        f.write(img_data["bytes"])
                    extract_count += 1

            records.append({
                "split": split_name,
                "filename": fname,
                "image_path": f"images/{fname}",
                "letters": row["letters"],
                "digits": row["digits"],
                "plate_text": f"{row['letters']} {row['digits']}" if pd.notna(row["letters"]) and pd.notna(row["digits"]) else str(row["digits"]),
                "province": row["province"],
                "province_id": row["label"],
                "color": row["color"],
                "reviewed": row["reviewed"],
                "source": row["source"],
            })

        print(f"Extracted {extract_count} images to {IMAGES_DIR}")

        split_df = pd.DataFrame(records)
        csv_path = LAO_DATASET_DIR / f"ground_truth_{split_name}.csv"
        split_df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Saved {split_name} ground truth to {csv_path}")

        dfs.append(split_df)

    # 4. Combined CSV
    combined_df = pd.concat(dfs, ignore_index=True)
    all_csv_path = LAO_DATASET_DIR / "ground_truth_all.csv"
    combined_df.to_csv(all_csv_path, index=False, encoding="utf-8")
    print(f"\nSaved combined ground truth (Total: {len(combined_df)} rows) to: {all_csv_path}")

    # 5. Province Distribution Summary
    print("\n--- Lao Province Distribution in Dataset ---")
    prov_counts = combined_df["province"].value_counts()
    print(prov_counts.to_string())

if __name__ == "__main__":
    download_and_extract_lao_dataset()
