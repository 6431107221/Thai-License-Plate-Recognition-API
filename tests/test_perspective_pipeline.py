"""
tests/test_perspective_pipeline.py

Unit and regression tests for 4-point polygon perspective transform cropping,
component extraction, and ground truth CSV generation pipeline.
Built with standard library `unittest` for zero-dependency execution.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.prepare_perspective_dataset import (
    extract_plate_components,
    order_points_clockwise,
    parse_polygon_coords,
    process_split,
    run_pipeline,
    warp_perspective_plate,
)


class TestPolygonParsing(unittest.TestCase):
    def test_parse_valid_polygon(self):
        line = "0 0.1 0.2 0.8 0.2 0.8 0.5 0.1 0.5"
        cls_id, pts = parse_polygon_coords(line, img_w=1000, img_h=500)
        self.assertEqual(cls_id, 0)
        self.assertEqual(pts.shape, (4, 2))
        np.testing.assert_allclose(pts[0], [100.0, 100.0])
        np.testing.assert_allclose(pts[1], [800.0, 100.0])
        np.testing.assert_allclose(pts[2], [800.0, 250.0])
        np.testing.assert_allclose(pts[3], [100.0, 250.0])

    def test_parse_valid_bbox(self):
        line = "0 0.5 0.5 0.2 0.1"
        cls_id, pts = parse_polygon_coords(line, img_w=100, img_h=100)
        self.assertEqual(cls_id, 0)
        self.assertEqual(pts.shape, (4, 2))
        np.testing.assert_allclose(pts[0], [40.0, 45.0])  # TL
        np.testing.assert_allclose(pts[1], [60.0, 45.0])  # TR
        np.testing.assert_allclose(pts[2], [60.0, 55.0])  # BR
        np.testing.assert_allclose(pts[3], [40.0, 55.0])  # BL

    def test_parse_too_few_coords_raises(self):
        line = "0 0.1 0.2 0.8"  # Only 3 coordinates
        with self.assertRaises(ValueError):
            parse_polygon_coords(line, img_w=100, img_h=100)

    def test_parse_empty_line_raises(self):
        with self.assertRaises(ValueError):
            parse_polygon_coords("   ", img_w=100, img_h=100)


class TestPointOrdering(unittest.TestCase):
    def test_order_clockwise_quadrilateral(self):
        # Shuffled 4 points: BR, TL, BL, TR
        pts = np.array([
            [800.0, 250.0],  # BR
            [100.0, 100.0],  # TL
            [100.0, 250.0],  # BL
            [800.0, 100.0],  # TR
        ], dtype=np.float32)

        ordered = order_points_clockwise(pts)
        np.testing.assert_allclose(ordered[0], [100.0, 100.0])  # TL
        np.testing.assert_allclose(ordered[1], [800.0, 100.0])  # TR
        np.testing.assert_allclose(ordered[2], [800.0, 250.0])  # BR
        np.testing.assert_allclose(ordered[3], [100.0, 250.0])  # BL

    def test_order_polygon_with_more_than_4_points(self):
        pts = np.array([
            [100.0, 100.0],
            [450.0, 100.0],
            [800.0, 100.0],
            [800.0, 250.0],
            [450.0, 250.0],
            [100.0, 250.0],
        ], dtype=np.float32)

        ordered = order_points_clockwise(pts)
        self.assertEqual(ordered.shape, (4, 2))


class TestPerspectiveWarp(unittest.TestCase):
    def test_warp_perspective_dimensions(self):
        canvas = np.zeros((400, 600, 3), dtype=np.uint8)
        pts = np.array([
            [100.0, 150.0],  # TL
            [350.0, 120.0],  # TR
            [360.0, 260.0],  # BR
            [110.0, 290.0],  # BL
        ], dtype=np.float32)

        warped = warp_perspective_plate(canvas, pts)
        self.assertIsNotNone(warped)
        self.assertEqual(warped.ndim, 3)
        h, w = warped.shape[:2]
        self.assertGreater(w, 200)
        self.assertGreater(h, 100)

    def test_warp_perspective_with_fixed_size(self):
        canvas = np.zeros((300, 300, 3), dtype=np.uint8)
        pts = np.array([
            [50.0, 50.0],
            [200.0, 50.0],
            [200.0, 120.0],
            [50.0, 120.0],
        ], dtype=np.float32)

        warped = warp_perspective_plate(canvas, pts, target_width=256, target_height=64)
        self.assertEqual(warped.shape, (64, 256, 3))


class TestComponentExtraction(unittest.TestCase):
    def test_proportional_component_split(self):
        plate_img = np.ones((100, 200, 3), dtype=np.uint8)
        plate_crop, prov_crop = extract_plate_components(plate_img, model_comp=None)

        self.assertEqual(plate_crop.shape[0], 65)
        self.assertEqual(plate_crop.shape[1], 200)
        self.assertEqual(prov_crop.shape[0], 40)
        self.assertEqual(prov_crop.shape[1], 200)


class TestPipelineEndToEnd(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.dataset_dir = self.temp_dir / "mock_yolo_dataset"
        self.output_dir = self.temp_dir / "output_crops"

        split_img_dir = self.dataset_dir / "train" / "images"
        split_lbl_dir = self.dataset_dir / "train" / "labels"
        split_img_dir.mkdir(parents=True)
        split_lbl_dir.mkdir(parents=True)

        # Create dummy image
        img = np.full((300, 400, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(split_img_dir / "car_sample_01.jpg"), img)

        # Create dummy polygon label
        label_content = "0 0.1 0.2 0.7 0.25 0.68 0.55 0.08 0.50\n"
        with open(split_lbl_dir / "car_sample_01.txt", "w", encoding="utf-8") as f:
            f.write(label_content)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_full_run(self):
        result = run_pipeline(
            dataset_dir=self.dataset_dir,
            output_dir=self.output_dir,
            splits=["train"],
            mode="full",
            tag_start=1,
        )

        self.assertEqual(result["total_records"], 1)
        self.assertEqual(result["next_tag"], 2)

        # Check directory structure
        train_out = self.output_dir / "train"
        self.assertTrue((train_out / "rectified_plates").exists())
        self.assertTrue((train_out / "plates").exists())
        self.assertTrue((train_out / "provs").exists())

        # Check file naming with tag index
        plate_files = list((train_out / "plates").glob("*.jpg"))
        prov_files = list((train_out / "provs").glob("*.jpg"))
        rect_files = list((train_out / "rectified_plates").glob("*.jpg"))

        self.assertEqual(len(plate_files), 1)
        self.assertEqual(len(prov_files), 1)
        self.assertEqual(len(rect_files), 1)

        self.assertTrue(plate_files[0].name.startswith("000001_car_sample_01_plate.jpg"))
        self.assertTrue(prov_files[0].name.startswith("000001_car_sample_01_prov.jpg"))
        self.assertTrue(rect_files[0].name.startswith("000001_car_sample_01_rectified.jpg"))

        # Check CSV
        csv_file = train_out / "train_unified.csv"
        self.assertTrue(csv_file.exists())

        df = pd.read_csv(csv_file)
        self.assertEqual(len(df), 1)
        self.assertEqual(
            list(df.columns),
            [
                "tag_id",
                "image",
                "prov_image",
                "rectified_image",
                "gt_plate",
                "gt_province",
                "original_image",
                "split",
            ],
        )
        row = df.iloc[0]
        self.assertEqual(row["tag_id"], 1)
        self.assertEqual(row["original_image"], "car_sample_01.jpg")
        self.assertEqual(row["split"], "train")

        # Verify compatibility with train_ocr.py path replacement rule:
        computed_prov_path = row["image"].replace("/plates/", "/provs/").replace("_plate", "_prov")
        self.assertEqual(computed_prov_path, row["prov_image"])
        self.assertTrue((self.output_dir / computed_prov_path).exists())


if __name__ == "__main__":
    unittest.main()
