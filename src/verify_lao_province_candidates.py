"""
src/verify_lao_province_candidates.py

Generates visual verification montages and an interactive HTML report
for inspecting harvested Lao province crops in datasets/Lao/lao_province_candidates/.

Output:
  - Montages saved to: output/lao_province_verification/montages/
  - HTML report: output/lao_province_verification/report.html
"""

import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CANDIDATES_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao_province_candidates"
CSV_PATH = CANDIDATES_DIR / "candidates_summary.csv"
OUTPUT_VERIFY_DIR = PROJECT_ROOT / "output" / "lao_province_verification"
MONTAGES_DIR = OUTPUT_VERIFY_DIR / "montages"
MAP_PATH = PROJECT_ROOT / "weights" / "province_map_lao.json"


def create_contact_sheet(image_paths: list, thumb_w: int = 200, thumb_h: int = 50, max_samples: int = 24) -> np.ndarray:
    """Generates an orderly visual contact sheet montage of crop images."""
    if not image_paths:
        return None

    # Sample evenly
    if len(image_paths) > max_samples:
        indices = np.linspace(0, len(image_paths) - 1, max_samples, dtype=int)
        selected = [image_paths[i] for i in indices]
    else:
        selected = image_paths

    n = len(selected)
    cols = 4
    rows = int(np.ceil(n / cols))

    pad = 4
    canvas_h = rows * thumb_h + (rows + 1) * pad
    canvas_w = cols * thumb_w + (cols + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    for i, p in enumerate(selected):
        r = i // cols
        c = i % cols
        img = cv2.imread(str(p))
        if img is None:
            continue
        resized = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        y = pad + r * (thumb_h + pad)
        x = pad + c * (thumb_w + pad)
        canvas[y : y + thumb_h, x : x + thumb_w] = resized

    return canvas


def generate_verification_report():
    print("=" * 70)
    print("   Generating Lao Province Visual Verification Report")
    print("=" * 70)

    OUTPUT_VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    MONTAGES_DIR.mkdir(parents=True, exist_ok=True)

    with open(MAP_PATH, "r", encoding="utf-8") as f:
        prov_map = json.load(f)

    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} does not exist. Please run src.harvest_lao_province_candidates first.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} total harvested candidate records from CSV.")

    cards_html = []
    summary_stats = []

    # Process each province
    for pid_str, pname in prov_map.items():
        pid = int(pid_str)
        folder_name = f"{pid:02d}_{pname}"
        folder_path = CANDIDATES_DIR / folder_name
        crop_files = sorted(list(folder_path.glob("*.jpg")))
        count = len(crop_files)

        sub_df = df[df["assigned_folder"] == pname] if "assigned_folder" in df.columns else pd.DataFrame()
        avg_conf = sub_df["confidence"].mean() if len(sub_df) > 0 else 0.0

        montage_rel = None
        if count > 0:
            montage_img = create_contact_sheet(crop_files)
            if montage_img is not None:
                montage_fn = f"montage_{folder_name}.jpg"
                montage_p = MONTAGES_DIR / montage_fn
                cv2.imwrite(str(montage_p), montage_img)
                montage_rel = f"montages/{montage_fn}"

        summary_stats.append({
            "pid": pid,
            "name": pname,
            "folder": folder_name,
            "count": count,
            "avg_conf": avg_conf,
            "montage": montage_rel,
        })

    # Also handle _low_confidence
    low_conf_folder = CANDIDATES_DIR / "_low_confidence"
    low_conf_files = sorted(list(low_conf_folder.glob("*.jpg")))
    low_count = len(low_conf_files)
    low_montage_rel = None
    if low_count > 0:
        low_montage = create_contact_sheet(low_conf_files)
        if low_montage is not None:
            low_fn = "montage_low_confidence.jpg"
            cv2.imwrite(str(MONTAGES_DIR / low_fn), low_montage)
            low_montage_rel = f"montages/{low_fn}"

    # Build HTML content
    for s in summary_stats:
        status_badge = '<span class="badge badge-success">Sufficient</span>' if s["count"] >= 30 else '<span class="badge badge-warning">Minority</span>'
        montage_tag = f'<img class="montage-img" src="{s["montage"]}" alt="{s["name"]}">' if s["montage"] else '<div class="no-crops">No crops harvested</div>'

        card = f"""
        <div class="prov-card">
          <div class="prov-card-header">
            <div class="prov-info">
              <span class="prov-id">#{s["pid"]:02d}</span>
              <h3 class="prov-name">{s["name"]}</h3>
            </div>
            <div class="prov-meta">
              <span class="count-pill">{s["count"]} crops</span>
              <span class="conf-pill">Avg: {s["avg_conf"]:.1f}%</span>
              {status_badge}
            </div>
          </div>
          <div class="montage-container">
            {montage_tag}
          </div>
        </div>
        """
        cards_html.append(card)

    low_card = f"""
    <div class="prov-card card-low-conf">
      <div class="prov-card-header">
        <div class="prov-info">
          <span class="prov-id">⚠️</span>
          <h3 class="prov-name">Low Confidence (< 35%)</h3>
        </div>
        <div class="prov-meta">
          <span class="count-pill pill-amber">{low_count} crops</span>
          <span class="badge badge-warning">Review Needed</span>
        </div>
      </div>
      <div class="montage-container">
        {f'<img class="montage-img" src="{low_montage_rel}">' if low_montage_rel else '<div class="no-crops">None</div>'}
      </div>
    </div>
    """
    cards_html.append(low_card)

    total_harvested = len(df)
    confident_harvested = len(df[df["is_confident"] == True]) if "is_confident" in df.columns else 0

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Lao Province Candidates - Visual Verification Dashboard</title>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #111827;
      --border: rgba(255, 255, 255, 0.1);
      --cyan: #00f0ff;
      --green: #10b981;
      --amber: #f59e0b;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0;
      padding: 32px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 24px;
      color: var(--cyan);
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 14px;
      margin: 0;
    }}
    .stats-bar {{
      display: flex;
      gap: 16px;
      margin-bottom: 28px;
    }}
    .stat-box {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px 20px;
      flex: 1;
    }}
    .stat-box .val {{
      font-size: 28px;
      font-weight: 700;
      color: var(--cyan);
    }}
    .stat-box .lbl {{
      font-size: 12px;
      color: var(--text-muted);
      text-transform: uppercase;
      margin-top: 4px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
      gap: 20px;
    }}
    .prov-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .prov-card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .prov-info {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .prov-id {{
      font-family: monospace;
      font-size: 12px;
      background: rgba(0, 240, 255, 0.15);
      color: var(--cyan);
      padding: 3px 8px;
      border-radius: 4px;
      font-weight: 600;
    }}
    .prov-name {{
      margin: 0;
      font-size: 18px;
    }}
    .prov-meta {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .count-pill {{
      font-size: 12px;
      background: rgba(255, 255, 255, 0.08);
      padding: 4px 10px;
      border-radius: 20px;
      font-weight: 600;
    }}
    .conf-pill {{
      font-size: 12px;
      color: var(--cyan);
      background: rgba(0, 240, 255, 0.08);
      padding: 4px 10px;
      border-radius: 20px;
      font-family: monospace;
    }}
    .pill-amber {{
      color: var(--amber);
      background: rgba(245, 158, 11, 0.15);
    }}
    .badge {{
      font-size: 10px;
      padding: 3px 8px;
      border-radius: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }}
    .badge-success {{
      background: rgba(16, 185, 129, 0.15);
      color: var(--green);
    }}
    .badge-warning {{
      background: rgba(245, 158, 11, 0.15);
      color: var(--amber);
    }}
    .montage-container {{
      border-radius: 8px;
      overflow: hidden;
      background: #000;
      display: flex;
      justify-content: center;
    }}
    .montage-img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 8px;
    }}
    .no-crops {{
      color: var(--text-muted);
      padding: 32px;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>🇱🇦 Lao Province Candidates - Visual Verification Dashboard</h1>
      <p class="subtitle">Inspecting separated province banner crops from 3 external datasets before retraining</p>
    </div>
  </div>

  <div class="stats-bar">
    <div class="stat-box">
      <div class="val">{total_harvested}</div>
      <div class="lbl">Total Crops Processed</div>
    </div>
    <div class="stat-box">
      <div class="val" style="color: var(--green);">{confident_harvested}</div>
      <div class="lbl">Confident Categorized (>= 35%)</div>
    </div>
    <div class="stat-box">
      <div class="val" style="color: var(--amber);">{low_count}</div>
      <div class="lbl">Review Required (Low Conf)</div>
    </div>
    <div class="stat-box">
      <div class="val">18</div>
      <div class="lbl">Lao Provinces Recognized</div>
    </div>
  </div>

  <div class="grid">
    {''.join(cards_html)}
  </div>
</body>
</html>
"""

    report_path = OUTPUT_VERIFY_DIR / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nVerification report generated at:")
    print(f"file://{report_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    generate_verification_report()
