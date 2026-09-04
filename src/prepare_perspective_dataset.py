"""
src/prepare_perspective_dataset.py

Pipeline for Thai License Plate 4-point polygon perspective transform cropping,
component separation (plate characters vs province), and ground truth CSV generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_annotation_coords(line: str, img_w: int, img_h: int) -> tuple[int, np.ndarray]:
    """Parses a YOLO annotation line (either 4-point polygon or standard bounding box) into
    class_id and pixel (x, y) coordinates.

    Format A (Polygon): class_id x1 y1 x2 y2 x3 y3 x4 y4 ... (normalized 0-1)
    Format B (BBox):    class_id x_center y_center width height (normalized 0-1)
    """
    parts = line.strip().split()
    if not parts:
        raise ValueError("Empty annotation line")

    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]

    if len(coords) == 4:
        # Standard YOLO bounding box -> convert to 4 corners
        x_c, y_c, bw, bh = coords
        x1 = max(0.0, (x_c - bw / 2.0) * img_w)
        y1 = max(0.0, (y_c - bh / 2.0) * img_h)
        x2 = min(float(img_w), (x_c + bw / 2.0) * img_w)
        y2 = min(float(img_h), (y_c + bh / 2.0) * img_h)
        pts = np.array([
            [x1, y1],  # TL
            [x2, y1],  # TR
            [x2, y2],  # BR
            [x1, y2],  # BL
        ], dtype=np.float32)
        return class_id, pts

    if len(coords) < 8:
        raise ValueError(f"Expected at least 4 bbox coords or 8 polygon coords, got {len(coords)}")

    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= img_w
    pts[:, 1] *= img_h
    return class_id, pts


# Backward compatibility alias
parse_polygon_coords = parse_annotation_coords


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """Ensures 4 points are ordered clockwise starting from top-left:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

    If pts has > 4 points, fits a minimum area bounding box.
    If pts already has 4 points in sequential order, validates and returns them.
    """
    if len(pts) > 4:
        rect = cv2.minAreaRect(pts.astype(np.float32))
        box = cv2.boxPoints(rect)
        pts = box

    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)  # y - x
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    unique_rows = np.unique(ordered, axis=0)
    if len(unique_rows) == 4:
        return ordered

    # Fallback for extreme tilt angles: sort by Y then by X
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]

    tl = top_two[np.argmin(top_two[:, 0])]
    tr = top_two[np.argmax(top_two[:, 0])]
    bl = bottom_two[np.argmin(bottom_two[:, 0])]
    br = bottom_two[np.argmax(bottom_two[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def is_valid_perspective_quad(pts: np.ndarray) -> bool:
    """Validates that 4 points form a physically plausible perspective quadrilateral of a license plate:
    - 4 distinct points
    - Strictly convex
    - Opposite side length ratios < 1.6 (perspective doesn't cause >60% difference between opposite edges)
    """
    if len(pts) != 4:
        return False
    pts_int = pts.astype(np.int32)
    if not cv2.isContourConvex(pts_int):
        return False
    if len(np.unique(pts_int, axis=0)) < 4:
        return False

    ordered = order_points_clockwise(pts)
    tl, tr, br, bl = ordered
    w_top = float(np.linalg.norm(tr - tl))
    w_bot = float(np.linalg.norm(br - bl))
    h_left = float(np.linalg.norm(bl - tl))
    h_right = float(np.linalg.norm(br - tr))

    if min(w_top, w_bot) <= 1e-3 or min(h_left, h_right) <= 1e-3:
        return False
    if max(w_top, w_bot) / min(w_top, w_bot) > 1.6:
        return False
    if max(h_left, h_right) / min(h_left, h_right) > 1.6:
        return False
    return True


def extract_quad_corners(
    mask: np.ndarray,
    img: np.ndarray | None = None,
    pad_frac: float = 0.08,
) -> np.ndarray:
    """Finds the 4 physical vertices of a license plate quadrilateral from a segmentation mask.
    Uses the minimum bounding oriented rectangle (minAreaRect) with margin padding.
    If the mask aspect ratio is unusually tall (< 1.7) due to a bumper protrusion (e.g. painted truck flowers),
    it isolates the high-contrast plate region using horizontal edge energy.
    """
    pts = mask.astype(np.float32).reshape(-1, 2)
    if len(pts) < 4:
        hull = pts
    else:
        hull = cv2.convexHull(pts).reshape(-1, 2)

    rect = cv2.minAreaRect(hull)
    center, (bw, bh), ang = rect
    if bw < bh:
        bw, bh = bh, bw
        ang += 90.0

    # Check for bumper appendages / vertically bloated masks (e.g. 0004 painted truck bumper art)
    ar = bw / max(bh, 1.0)
    if ar < 1.7 and img is not None:
        target_h = bw / 2.25
        raw_pts = cv2.boxPoints(rect).astype(np.float32)
        raw_crop = warp_perspective_plate(img, raw_pts, padding_frac=0.0)
        if raw_crop.shape[0] > 10:
            gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
            grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
            row_energy = grad_y.sum(axis=1)
            win_sz = int(round(target_h / bh * raw_crop.shape[0]))
            win_sz = max(min(win_sz, len(row_energy)), 1)
            best_e = -1.0
            best_offset = 0
            for offset in range(len(row_energy) - win_sz + 1):
                e = float(row_energy[offset:offset + win_sz].sum())
                if e > best_e:
                    best_e = e
                    best_offset = offset

            frac_shift = (best_offset + win_sz / 2.0) / raw_crop.shape[0] - 0.5
            rad = np.radians(ang + 90.0)
            dx = frac_shift * bh * np.cos(rad)
            dy = frac_shift * bh * np.sin(rad)
            center = (center[0] + dx, center[1] + dy)
            bh = target_h

    rect_padded = (center, (bw * (1.0 + pad_frac), bh * (1.0 + pad_frac)), ang)
    return cv2.boxPoints(rect_padded).astype(np.float32)


def fine_deskew_plate(
    plate_img: np.ndarray,
    max_angle: float = 8.0,
    step: float = 0.25,
) -> np.ndarray:
    """Automatically detects remaining tilt (from plate borders and character baselines)
    using horizontal projection profile variance, and rotates the image so text and
    borders are strictly parallel to the horizontal axis.

    This avoids being fooled by diagonal digit strokes ('4', '6', '7') or crop boundaries
    that break naive Hough line detection.
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img

    h, w = plate_img.shape[:2]
    if h < 16 or w < 16:
        return plate_img

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Focus on the inner plate area (excluding outer 8% padding to avoid bumper/crop edges)
    my = max(int(h * 0.08), 2)
    mx = max(int(w * 0.06), 2)
    roi = gray[my:h - my, mx:w - mx]
    rh, rw = roi.shape
    if rh < 10 or rw < 10:
        return plate_img

    # Horizontal gradient (Sobel Y highlights horizontal lines: text baselines and plate borders)
    grad_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    abs_grad = np.abs(grad_y)
    grad_max = float(abs_grad.max())
    if grad_max <= 1e-5:
        return plate_img

    abs_grad = (abs_grad / grad_max * 255.0).astype(np.uint8)
    _, binary_grad = cv2.threshold(abs_grad, 40, 255, cv2.THRESH_BINARY)

    angles = np.arange(-max_angle, max_angle + step / 2.0, step)
    variances = []
    center_roi = (rw / 2.0, rh / 2.0)

    for ang in angles:
        M = cv2.getRotationMatrix2D(center_roi, float(ang), 1.0)
        rot = cv2.warpAffine(
            binary_grad, M, (rw, rh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
        )
        row_sums = rot.sum(axis=1).astype(np.float64)
        variances.append(float(np.var(row_sums)))

    best_idx = int(np.argmax(variances))
    best_ang = float(angles[best_idx])

    if abs(best_ang) < 0.2:  # Negligible tilt
        return plate_img

    # Rotate original crop to align text and borders with horizontal axis
    center = (w / 2.0, h / 2.0)
    M_crop = cv2.getRotationMatrix2D(center, best_ang, 1.0)
    deskewed = cv2.warpAffine(
        plate_img, M_crop, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed


def warp_perspective_plate(
    image: np.ndarray,
    pts: np.ndarray,
    target_width: int | None = None,
    target_height: int | None = None,
    min_dimension: int = 16,
    padding_frac: float = 0.0,
) -> np.ndarray:
    """Warps a quadrilateral license plate region into a front-facing rectangle.

    Args:
        padding_frac: Fraction of the plate width/height to expand the quad corners outward
                      before warping. Helps capture edges that the model slightly underestimates.
    """
    ordered_pts = order_points_clockwise(pts)
    tl, tr, br, bl = ordered_pts

    if padding_frac > 0:
        # Expand each corner outward by padding_frac of the plate dimensions
        cx = float(np.mean([tl[0], tr[0], br[0], bl[0]]))
        cy = float(np.mean([tl[1], tr[1], br[1], bl[1]]))
        expanded = []
        for pt in [tl, tr, br, bl]:
            dx = pt[0] - cx
            dy = pt[1] - cy
            norm = max(np.sqrt(dx ** 2 + dy ** 2), 1e-6)
            w_top = np.linalg.norm(tr - tl)
            h_left = np.linalg.norm(bl - tl)
            pad = padding_frac * max(w_top, h_left)
            expanded.append([pt[0] + dx / norm * pad, pt[1] + dy / norm * pad])
        tl, tr, br, bl = [np.array(p, dtype=np.float32) for p in expanded]
        ordered_pts = np.array([tl, tr, br, bl], dtype=np.float32)

    if target_width is None or target_height is None:
        w_top = np.linalg.norm(tr - tl)
        w_bot = np.linalg.norm(br - bl)
        calc_w = max(int(max(w_top, w_bot)), min_dimension)

        h_left = np.linalg.norm(bl - tl)
        h_right = np.linalg.norm(br - tr)
        calc_h = max(int(max(h_left, h_right)), min_dimension)
    else:
        calc_w = max(target_width, min_dimension)
        calc_h = max(target_height, min_dimension)

    dst_pts = np.array(
        [
            [0, 0],
            [calc_w - 1, 0],
            [calc_w - 1, calc_h - 1],
            [0, calc_h - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (calc_w, calc_h), flags=cv2.INTER_CUBIC)
    return warped


def extract_plate_components(
    rectified_plate: np.ndarray,
    model_comp: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Splits a rectified plate into plate characters crop and province crop.

    If model_comp (YOLO component detector) is provided, runs detection.
    Otherwise, applies standard proportional fallback:
    - Plate characters: top 65% of plate
    - Province: bottom 40% of plate (from y=60% to 100%)
    """
    h, w = rectified_plate.shape[:2]
    plate_crop = None
    prov_crop = None

    if model_comp is not None:
        try:
            results = model_comp(rectified_plate, verbose=False)[0]
            for box in results.boxes:
                cls_idx = int(box.cls[0])
                cls_name = model_comp.names[cls_idx]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                comp_img = rectified_plate[y1:y2, x1:x2]
                if comp_img.size == 0:
                    continue

                cls_lower = cls_name.lower()
                if "plate" in cls_lower or "char" in cls_lower:
                    plate_crop = comp_img
                elif "prov" in cls_lower:
                    prov_crop = comp_img
        except Exception as err:
            print(f"   [Warning] Model 2 inference failed: {err}. Using proportional fallback.")

    if plate_crop is None or plate_crop.size == 0:
        plate_crop = rectified_plate[0 : int(h * 0.65), 0:w]

    if prov_crop is None or prov_crop.size == 0:
        prov_crop = rectified_plate[int(h * 0.60) : h, 0:w]

    return plate_crop, prov_crop


def resolve_split_dirs(dataset_root: Path, split_name: str) -> tuple[Path, Path] | None:
    """Intelligently locates images and labels directories for a given split across
    common YOLO dataset directory conventions:
    1. Direct image directory: dataset_root is already images/<split>
    2. Convention A: dataset_root/<split>/images and dataset_root/<split>/labels
    3. Convention B: dataset_root/images/<split> and dataset_root/labels/<split>
    4. Convention C: dataset_root/images and dataset_root/labels
    """
    # 1. Direct path to images/<split>
    if dataset_root.is_dir():
        if dataset_root.name == split_name and dataset_root.parent.name == "images":
            candidate_lbl = dataset_root.parent.parent / "labels" / split_name
            if candidate_lbl.exists():
                return dataset_root, candidate_lbl
        elif dataset_root.name == "images" and (dataset_root / split_name).exists():
            candidate_lbl = dataset_root.parent / "labels" / split_name
            if candidate_lbl.exists():
                return dataset_root / split_name, candidate_lbl

    # 2. Convention A: root/<split>/images
    cand_img_a = dataset_root / split_name / "images"
    cand_lbl_a = dataset_root / split_name / "labels"
    if cand_img_a.exists() and cand_lbl_a.exists():
        return cand_img_a, cand_lbl_a

    # 3. Convention B: root/images/<split>
    cand_img_b = dataset_root / "images" / split_name
    cand_lbl_b = dataset_root / "labels" / split_name
    if cand_img_b.exists() and cand_lbl_b.exists():
        return cand_img_b, cand_lbl_b

    # 4. Single-folder dataset without splits
    cand_img_c = dataset_root / "images"
    cand_lbl_c = dataset_root / "labels"
    if cand_img_c.exists() and cand_lbl_c.exists():
        return cand_img_c, cand_lbl_c

    # Fallback check if dataset_root itself contains images and labels
    if (dataset_root / "labels").exists():
        return dataset_root, dataset_root / "labels"

    return None


def merge_overlapping_masks(
    masks: list[np.ndarray],
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    iou_thresh: float = 0.35,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Merges overlapping segmentation masks that belong to the same license plate.
    This prevents plates being split into left/right fragments (e.g. cutting off edge characters).
    """
    n = len(masks)
    if n <= 1:
        return [(masks[i], boxes_xyxy[i], float(confs[i])) for i in range(n)]

    merged: list[tuple[np.ndarray, np.ndarray, float]] = []
    used = [False] * n
    for i in range(n):
        if used[i]:
            continue
        curr_m = masks[i]
        curr_b = list(boxes_xyxy[i])
        max_c = float(confs[i])
        for j in range(i + 1, n):
            if used[j]:
                continue
            b1 = curr_b
            b2 = boxes_xyxy[j]
            ix1 = max(b1[0], b2[0])
            iy1 = max(b1[1], b2[1])
            ix2 = min(b1[2], b2[2])
            iy2 = min(b1[3], b2[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union = a1 + a2 - inter
            iou = inter / union if union > 0 else 0.0
            if iou > iou_thresh:
                curr_m = np.vstack([curr_m, masks[j]])
                curr_b = [
                    min(b1[0], b2[0]),
                    min(b1[1], b2[1]),
                    max(b1[2], b2[2]),
                    max(b1[3], b2[3]),
                ]
                max_c = max(max_c, float(confs[j]))
                used[j] = True
        merged.append((curr_m, np.array(curr_b, dtype=np.float32), max_c))
        used[i] = True
    return merged


def process_split(
    dataset_root: Path,
    output_root: Path,
    split_name: str,
    mode: str = "full",
    model_comp: Any | None = None,
    model_plate_seg: Any | None = None,
    conf: float = 0.25,
    tag_start: int = 1,
    target_width: int | None = None,
    target_height: int | None = None,
    allowed_classes: list[int] | None = None,
    padding_frac: float | None = None,
    max_plates_per_image: int | None = 1,
) -> tuple[list[dict[str, Any]], int]:
    """Processes a single split (train/valid/test) of a YOLO polygon or bbox dataset."""
    dirs = resolve_split_dirs(dataset_root, split_name)
    if dirs is None:
        print(f"Skipping split '{split_name}': directory not found under {dataset_root}.")
        return [], tag_start

    img_dir, lbl_dir = dirs

    rectified_dir = output_root / split_name / "rectified_plates"
    plates_dir = output_root / split_name / "plates"
    provs_dir = output_root / split_name / "provs"

    rectified_dir.mkdir(parents=True, exist_ok=True)
    if mode in ("full", "components"):
        plates_dir.mkdir(parents=True, exist_ok=True)
        provs_dir.mkdir(parents=True, exist_ok=True)

    img_files: list[Path] = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG"]:
        img_files.extend(list(img_dir.glob(ext)))
    img_files.sort(key=lambda p: p.name)

    print(f"\nProcessing split '{split_name}' from {img_dir}: {len(img_files)} images found.")
    records: list[dict[str, Any]] = []
    current_tag = tag_start
    processed_count = 0

    for img_path in tqdm(img_files, desc=f"Split [{split_name}]"):
        lbl_path = lbl_dir / f"{img_path.stem}.txt" if lbl_dir else None

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        detected_polygons: list[np.ndarray] = []

        if model_plate_seg is not None:
            try:
                results = model_plate_seg.predict(source=img, conf=conf, verbose=False)[0]
                if results.masks is not None and len(results.masks.xy) > 0:
                    raw_masks = [m for m in results.masks.xy]
                    raw_boxes = results.boxes.xyxy.cpu().numpy()
                    raw_confs = results.boxes.conf.cpu().numpy()

                    # 1. Merge overlapping masks (IoU > 0.35) so fragmented plates are unified
                    merged_dets = merge_overlapping_masks(raw_masks, raw_boxes, raw_confs, iou_thresh=0.35)

                    # 2. Score candidates using Thai plate aspect ratios
                    # Thai plates: Cars 340x150mm (AR ~2.27), Trucks 440x220mm (AR ~2.0), Motorbikes (AR ~1.28)
                    PLATE_AR_MIN = 1.1
                    PLATE_AR_MAX = 3.8
                    PLATE_AR_IDEAL = 2.2

                    def plate_score(box: np.ndarray, conf_val: float) -> float:
                        bw = float(box[2] - box[0])
                        bh = float(box[3] - box[1])
                        if bh < 1.0:
                            return 0.0
                        ar = bw / bh
                        if ar < PLATE_AR_MIN:
                            ar_penalty = (PLATE_AR_MIN - ar) / PLATE_AR_MIN
                        elif ar > PLATE_AR_MAX:
                            ar_penalty = (ar - PLATE_AR_MAX) / PLATE_AR_MAX
                        else:
                            ar_penalty = abs(ar - PLATE_AR_IDEAL) / (PLATE_AR_MAX - PLATE_AR_MIN) * 0.15
                        return float(conf_val) * (1.0 - min(ar_penalty, 1.0))

                    scored_dets = []
                    for m, b, c in merged_dets:
                        s = plate_score(b, c)
                        scored_dets.append((s, m, b, c))

                    # Sort by score descending (best plate match first)
                    scored_dets.sort(key=lambda x: x[0], reverse=True)

                    # 3. Limit to max_plates_per_image if specified (e.g. 1 for vehicle-cropped images)
                    if max_plates_per_image is not None and max_plates_per_image > 0:
                        scored_dets = scored_dets[:max_plates_per_image]

                    for s, mask, box, c in scored_dets:
                        if len(mask) < 3:
                            continue
                        quad_corners = extract_quad_corners(mask, img=img)
                        pts = order_points_clockwise(quad_corners)
                        detected_polygons.append(pts)
            except Exception as err:
                print(f"   [Error] Segmentation inference failed on {img_path.name}: {err}")
        elif lbl_path is not None and lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            for line in lines:
                try:
                    class_id, pts = parse_annotation_coords(line, w, h)
                    if allowed_classes is not None and class_id not in allowed_classes:
                        continue
                    detected_polygons.append(pts)
                except Exception:
                    continue
        else:
            continue

        plate_idx = 0
        effective_padding = (
            padding_frac if padding_frac is not None else 0.0
        )
        for pts in detected_polygons:
            try:
                rectified = warp_perspective_plate(
                    img,
                    pts,
                    target_width=target_width,
                    target_height=target_height,
                    padding_frac=effective_padding,
                )
                rectified = fine_deskew_plate(rectified)
            except Exception as err:
                print(f"   [Error] Perspective warp failed on {img_path.name}: {err}")
                continue

            if rectified.size == 0:
                continue

            base_name = img_path.stem
            suffix = f"_{plate_idx}" if len(detected_polygons) > 1 else ""

            # Save front-view rectified plate (for annotation)
            save_rect_name = f"{current_tag:06d}_{base_name}{suffix}_rectified.jpg"
            cv2.imwrite(str(rectified_dir / save_rect_name), rectified)

            # Component crops (plate character vs province)
            if mode in ("full", "components"):
                plate_crop, prov_crop = extract_plate_components(rectified, model_comp)

                save_plate_name = f"{current_tag:06d}_{base_name}{suffix}_plate.jpg"
                save_prov_name = f"{current_tag:06d}_{base_name}{suffix}_prov.jpg"

                cv2.imwrite(str(plates_dir / save_plate_name), plate_crop)
                cv2.imwrite(str(provs_dir / save_prov_name), prov_crop)

                # Unified CSV record format compatible with train_ocr.py and datasets.py
                records.append({
                    "tag_id": current_tag,
                    "image": f"{split_name}/plates/{save_plate_name}",
                    "prov_image": f"{split_name}/provs/{save_prov_name}",
                    "rectified_image": f"{split_name}/rectified_plates/{save_rect_name}",
                    "gt_plate": "",
                    "gt_province": "",
                    "original_image": img_path.name,
                    "split": split_name,
                })
            else:
                # Mode: rectify_only
                records.append({
                    "tag_id": current_tag,
                    "image": f"{split_name}/rectified_plates/{save_rect_name}",
                    "original_image": img_path.name,
                    "split": split_name,
                })

            current_tag += 1
            processed_count += 1
            plate_idx += 1

    print(f" Split '{split_name}' complete: {processed_count} plates processed.")
    return records, current_tag


def run_pipeline(
    dataset_dir: str | Path,
    output_dir: str | Path,
    splits: list[str] | None = None,
    mode: str = "full",
    model_comp_path: str | Path | None = None,
    model_plate_seg_path: str | Path | None = None,
    conf: float = 0.25,
    tag_start: int = 1,
    target_width: int | None = None,
    target_height: int | None = None,
    allowed_classes: list[int] | None = None,
    padding_frac: float | None = None,
    max_plates_per_image: int | None = 1,
) -> dict[str, Any]:
    """Runs the perspective transform and dataset preparation pipeline."""
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if splits is None:
        splits = ["train", "valid", "test"]

    model_comp = None
    if model_comp_path:
        comp_p = Path(model_comp_path).resolve()
        if comp_p.exists():
            from ultralytics import YOLO  # type: ignore

            print(f"Loading Model 2 (Component Detector) from {comp_p}...")
            model_comp = YOLO(str(comp_p))
        else:
            print(f"[Warning] Component model path {comp_p} not found. Using fallback.")

    model_plate_seg = None
    if model_plate_seg_path:
        seg_p = Path(model_plate_seg_path).resolve()
        if seg_p.exists():
            from ultralytics import YOLO  # type: ignore

            print(f"Loading Plate Polygon Segmentation Model from {seg_p}...")
            model_plate_seg = YOLO(str(seg_p))
        else:
            print(f"[Warning] Plate segmentation model path {seg_p} not found.")

    print(f"=== License Plate Dataset Preparation ===")
    print(f"Input Dataset:  {dataset_dir}")
    print(f"Output Root:    {output_dir}")
    print(f"Target Splits:  {splits}")
    print(f"Operating Mode: {mode}")

    current_tag = tag_start
    all_records: list[dict[str, Any]] = []

    # If dataset_dir points directly to a single split (e.g. images/train)
    if dataset_dir.name in ["train", "val", "valid", "test"]:
        effective_split = "valid" if dataset_dir.name == "val" else dataset_dir.name
        splits = [effective_split]

    for split in splits:
        records, current_tag = process_split(
            dataset_root=dataset_dir,
            output_root=output_dir,
            split_name=split,
            mode=mode,
            model_comp=model_comp,
            model_plate_seg=model_plate_seg,
            conf=conf,
            tag_start=current_tag,
            target_width=target_width,
            target_height=target_height,
            allowed_classes=allowed_classes,
            padding_frac=padding_frac,
            max_plates_per_image=max_plates_per_image,
        )

        if records:
            df = pd.DataFrame(records)
            csv_filename = f"{split}_unified.csv"
            out_csv_path = output_dir / split / csv_filename
            df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
            print(f" Saved split CSV: {out_csv_path} ({len(df)} rows)")
            all_records.extend(records)

    # Master summary CSV across all splits
    if all_records:
        master_df = pd.DataFrame(all_records)
        master_csv_path = output_dir / "all_unified.csv"
        master_df.to_csv(master_csv_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved Master CSV: {master_csv_path} ({len(master_df)} total records)")

    return {"total_records": len(all_records), "next_tag": current_tag}


def main():
    parser = argparse.ArgumentParser(
        description="Thai License Plate 4-Point Perspective Transform & Ground Truth Preparation"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/LPR 2 - Polygon.yolov11",
        help="Path to input YOLO polygon dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Path to output directory for crops and CSV files (default: output)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Dataset splits to process (e.g., train valid test)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "rectify", "components"],
        default="full",
        help="Operation mode: 'full' (rectify + components + csv), 'rectify' (front view only), or 'components'",
    )
    parser.add_argument(
        "--model-comp",
        type=str,
        default=None,
        help="Path to trained Model 2 YOLO component detector weights (.pt)",
    )
    parser.add_argument(
        "--model-plate-seg",
        type=str,
        default=None,
        help="Path to trained YOLO plate polygon/segmentation model (.pt) to detect tilted plates and perspective-warp them automatically from raw car images",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO models (default: 0.25)",
    )
    parser.add_argument(
        "--allowed-classes",
        type=int,
        nargs="+",
        default=None,
        help="Allowed class IDs to crop (default: all classes in dataset)",
    )
    parser.add_argument(
        "--tag-start",
        type=int,
        default=1,
        help="Initial integer for sequential tag indexing",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=None,
        help="Optional fixed target width for perspective transform (default: dynamic)",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=None,
        help="Optional fixed target height for perspective transform (default: dynamic)",
    )
    parser.add_argument(
        "--padding-frac",
        type=float,
        default=None,
        help="Fraction to expand corners outward before warping (default: 0.08 for model seg, 0.0 for labels)",
    )
    parser.add_argument(
        "--max-plates-per-image",
        type=int,
        default=1,
        help="Maximum plates to extract per image (default: 1 for vehicle crops; 0 for all)",
    )

    args = parser.parse_args()
    max_plates = args.max_plates_per_image if args.max_plates_per_image > 0 else None
    run_pipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        splits=args.splits,
        mode=args.mode,
        model_comp_path=args.model_comp,
        model_plate_seg_path=args.model_plate_seg,
        conf=args.conf,
        tag_start=args.tag_start,
        target_width=args.target_width,
        target_height=args.target_height,
        allowed_classes=args.allowed_classes,
        padding_frac=args.padding_frac,
        max_plates_per_image=max_plates,
    )


if __name__ == "__main__":
    main()
