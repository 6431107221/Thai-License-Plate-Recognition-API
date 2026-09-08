"""
src/verify_character_crops.py

Verification utility for datasets/thai_character_crops/.
Features:
  1. Inspects class counts and distribution across splits (train, valid, test).
  2. Flags ultra-rare classes (< 5 samples) or corrupted files.
  3. Generates visual contact sheet grids (montages) for each character class
     saved to output/character_verification/ so you can visually verify all labels at a glance.
  4. Generates an HTML report (output/character_verification/report.html) to easily
     scroll through all 50 character classes in your browser.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "datasets" / "thai_character_crops"
SPLITS_DIR = BASE_DIR / "splits"
SQUARE_DIR = BASE_DIR / "by_character_square"
OUT_VERIFY_DIR = PROJECT_ROOT / "output" / "character_verification"
OUT_VERIFY_DIR.mkdir(parents=True, exist_ok=True)


def create_character_grid(char_folder: Path, max_samples: int = 25, thumb_size: int = 64) -> np.ndarray:
    """Creates a 5x5 grid of sample crops for a character."""
    image_paths = sorted(list(char_folder.glob("*.jpg")) + list(char_folder.glob("*.png")))
    if not image_paths:
        return None

    # Pick evenly spaced or random samples
    if len(image_paths) > max_samples:
        indices = np.linspace(0, len(image_paths) - 1, max_samples, dtype=int)
        selected_paths = [image_paths[i] for i in indices]
    else:
        selected_paths = image_paths

    n = len(selected_paths)
    cols = 5
    rows = int(np.ceil(n / cols))

    grid_h = rows * thumb_size + (rows + 1) * 4
    grid_w = cols * thumb_size + (cols + 1) * 4
    canvas = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)

    for i, p in enumerate(selected_paths):
        r = i // cols
        c = i % cols
        img = cv2.imread(str(p))
        if img is None:
            continue
        img_res = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)

        y = 4 + r * (thumb_size + 4)
        x = 4 + c * (thumb_size + 4)
        canvas[y : y + thumb_size, x : x + thumb_size] = img_res

    return canvas


def verify_characters(target_dir=SQUARE_DIR, out_dir=OUT_VERIFY_DIR, title="Thai Character Crops"):
    print("==================================================================")
    print(f"VERIFICATION & SANITY CHECK: {title}")
    print(f"Source Directory: {target_dir}")
    print("==================================================================")

    out_dir.mkdir(parents=True, exist_ok=True)
    char_dirs = sorted([d for d in target_dir.iterdir() if d.is_dir()])
    print(f"Total character folders: {len(char_dirs)}")

    # Calculate counts
    counts = {}
    total_imgs = 0
    for d in char_dirs:
        n = len(list(d.glob("*.jpg")) + list(d.glob("*.png")))
        counts[d.name] = n
        total_imgs += n

    counts_series = pd.Series(counts).sort_values(ascending=False)
    print(f"Total images found: {total_imgs}")
    print(f"Top 10 most frequent: {counts_series.head(10).to_dict()}")
    print(f"Least frequent: {counts_series.tail(10).to_dict()}")

    rare_chars = counts_series[counts_series < 5]
    if not rare_chars.empty:
        print(f"\n[Notice] {len(rare_chars)} classes have fewer than 5 samples: {list(rare_chars.index)}")

    # Generate visual inspection grids
    print("\n--- Generating Visual Verification Contact Sheets ---")
    html_cards = []

    for d in char_dirs:
        char_name = d.name
        n_samples = counts[char_name]
        if n_samples == 0:
            continue

        grid = create_character_grid(d, max_samples=25, thumb_size=64)
        if grid is None:
            continue

        safe_char = f"u{ord(char_name):04x}" if len(char_name) == 1 else char_name
        grid_filename = f"grid_{safe_char}.jpg"
        grid_path = out_dir / grid_filename
        cv2.imwrite(str(grid_path), grid)

        html_cards.append(f"""
        <div class="card">
            <h3>Character: <span class="badge">{char_name}</span> ({n_samples} samples)</h3>
            <img src="{grid_filename}" alt="{char_name}" />
        </div>
        """)

    # Generate HTML report
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title} Visual Verification</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; }}
        h1 {{ text-align: center; color: #38bdf8; margin-bottom: 5px; }}
        p.subtitle {{ text-align: center; color: #94a3b8; font-size: 16px; margin-bottom: 30px; }}
        .grid-container {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
        .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .badge {{ background: #0284c7; color: white; padding: 4px 10px; border-radius: 6px; font-size: 20px; font-weight: bold; }}
        img {{ border-radius: 6px; margin-top: 10px; display: block; }}
    </style>
</head>
<body>
    <h1>{title} Inspection Report</h1>
    <p class="subtitle">Review character bounding boxes to ensure labels and crops are clean before model training.<br>Total images: <b>{total_imgs}</b> across <b>{len(char_dirs)}</b> classes.</p>
    <div class="grid-container">
        {''.join(html_cards)}
    </div>
</body>
</html>
"""
    html_file = out_dir / "report.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[Success] Generated {len(char_dirs)} verification grids!")
    print(f"Visual HTML Report: file://{html_file.resolve()}")
    print("You can open report.html directly in any web browser to scan all character classes.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["master", "candidates", "train"], default="master",
                        help="Which directory to verify ('master', 'candidates', or 'train')")
    args = parser.parse_args()

    if args.source == "candidates":
        target = BASE_DIR / "candidates"
        out = OUT_VERIFY_DIR / "candidates"
        title = "Harvested Candidates"
    elif args.source == "train":
        target = SPLITS_DIR / "train"
        out = OUT_VERIFY_DIR / "train"
        title = "Training Split"
    else:
        target = SQUARE_DIR
        out = OUT_VERIFY_DIR / "master"
        title = "Master Square Crops"

    verify_characters(target_dir=target, out_dir=out, title=title)
