"""Regression tests for non-destructive image adjustments and mosaic exports."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from SweepAtlas import AtlasTiles, build_display, SweepAtlas
from SweepAtlasAdjust import ImageAdjustments, HALO
from SweepAtlasVideo import video_display


class AdjustmentTests(unittest.TestCase):
    def test_defaults_do_not_change_source_pixels(self):
        image = np.arange(64 * 48 * 3, dtype=np.uint8).reshape(48, 64, 3)
        controls = ImageAdjustments()
        self.assertFalse(controls.active)
        self.assertIs(controls.apply(image), image)
        self.assertTrue(np.array_equal(controls.apply(image), image))

    def test_contrast_extremes_and_midpoint(self):
        image = np.array([[[64, 127, 192]]], dtype=np.uint8)
        stronger = ImageAdjustments(contrast=150).apply(image)
        self.assertLess(int(stronger[0, 0, 0]), 64)
        self.assertGreater(int(stronger[0, 0, 2]), 192)
        weaker = ImageAdjustments(contrast=50).apply(image)
        self.assertGreater(int(weaker[0, 0, 0]), 64)
        self.assertLess(int(weaker[0, 0, 2]), 192)

    def test_sharpness_changes_edges_not_flat_regions(self):
        image = np.full((32, 40, 3), 110, dtype=np.uint8)
        image[:, 20:] = 180
        sharpened = ImageAdjustments(sharpness=100).apply(image)
        self.assertEqual(tuple(sharpened[4, 4]), (110, 110, 110))
        self.assertLess(int(sharpened[12, 19, 0]), 110)
        self.assertGreater(int(sharpened[12, 20, 0]), 180)

    def test_mask_unchanged_background(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
        mask = np.zeros((40, 50), np.uint8)
        mask[5:32, 8:43] = 255
        adjusted = ImageAdjustments(contrast=180, sharpness=80).apply(image, mask)
        self.assertTrue(np.array_equal(adjusted[mask == 0], image[mask == 0]))
        self.assertFalse(np.array_equal(adjusted[mask != 0], image[mask != 0]))

    def test_striped_enhancement_matches_whole_image(self):
        rng = np.random.default_rng(33)
        bgra = rng.integers(0, 256, (71, 85, 4), dtype=np.uint8)
        bgra[:, :, 3] = 255
        bgra[11:32, 15:45, 3] = 0
        original = bgra.copy()
        controls = ImageAdjustments(contrast=140, sharpness=100)
        expected = controls.apply(original[:, :, :3], original[:, :, 3])
        controls.apply_rgba_inplace(bgra, stripe_height=HALO + 6)
        np.testing.assert_array_equal(bgra[:, :, :3], expected)
        np.testing.assert_array_equal(bgra[:, :, 3], original[:, :, 3])

    def test_file_export_alpha_and_unfiltered_source_are_preserved(self):
        rng = np.random.default_rng(78)
        source = rng.integers(0, 255, (88, 112, 3), dtype=np.uint8)
        tiles = AtlasTiles(tile_size=64)
        self.assertTrue(tiles.add(source, np.eye(3, dtype=np.float64)))
        stored = {key: rgb.copy() for key, (rgb, _) in tiles.tiles.items()}
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "original.png"
            tuned_path = Path(temp_dir) / "adjusted.png"
            tiles.save(raw_path, max_megapixels=5.)
            tiles.save(tuned_path, max_megapixels=5.,
                       adjustments=ImageAdjustments(contrast=175, sharpness=75))
            raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
            tuned = cv2.imread(str(tuned_path), cv2.IMREAD_UNCHANGED)
            self.assertEqual(raw.shape, tuned.shape)
            np.testing.assert_array_equal(raw[:, :, 3], tuned[:, :, 3])
            self.assertTrue(np.any(raw[:, :, 3] == 0))
            self.assertTrue(np.any(raw[:, :, 3] == 255))
            self.assertFalse(np.array_equal(raw[:, :, :3], tuned[:, :, :3]))
            for key, value in stored.items():
                np.testing.assert_array_equal(tiles.tiles[key][0], value)

    def test_video_and_webcam_displays_accept_adjustments(self):
        frame = np.full((48, 64, 3), (80, 120, 160), np.uint8)
        atlas = SweepAtlas()
        controls = ImageAdjustments(contrast=135, sharpness=60)
        webcam = build_display(frame, atlas, False, True, 25.,
                               adjustments=controls)
        video = video_display(frame, atlas, False, 25., "example.mp4", 1, 100,
                              "", adjustments=controls)
        self.assertEqual(webcam.shape, (550, 1520, 3))
        self.assertEqual(video.shape, (628, 1520, 3))


if __name__ == "__main__":
    unittest.main()
