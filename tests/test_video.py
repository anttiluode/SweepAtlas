"""Headless checks for SweepAtlas's video load, frame seeking, and timeline."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from SweepAtlas import SweepAtlas
from SweepAtlasVideo import (
    BAR_LEFT, BAR_RIGHT, frame_from_x, format_time, open_capture, read_at,
    video_display,
)


class VideoControlsTests(unittest.TestCase):
    def test_timeline_clamps_to_first_and_last_frame(self):
        self.assertEqual(frame_from_x(BAR_LEFT - 300, 101), 0)
        self.assertEqual(frame_from_x(BAR_RIGHT + 300, 101), 100)
        self.assertEqual(frame_from_x((BAR_LEFT + BAR_RIGHT) / 2, 101), 50)
        self.assertEqual(frame_from_x(BAR_LEFT, 0), 0)

    def test_clock_formatting(self):
        self.assertEqual(format_time(0), "00:00:00")
        self.assertEqual(format_time(3661), "01:01:01")

    def test_missing_video_is_error(self):
        with self.assertRaises(OSError):
            open_capture(Path("this_video_does_not_exist_92385.mp4"))

    def test_seek_reads_requested_part_of_video(self):
        with tempfile.TemporaryDirectory() as td:
            file = Path(td) / "seek.avi"
            writer = cv2.VideoWriter(str(file), cv2.VideoWriter_fourcc(*"MJPG"),
                                     12.0, (160, 120))
            if not writer.isOpened():
                self.skipTest("MJPEG video encoder unavailable")
            for i in range(16):
                image = np.full((120, 160, 3), (10, 15, 20), dtype=np.uint8)
                image[:, :, 0] = i * 12
                cv2.circle(image, (30+i*5, 60), 12, (255, 255, 255), -1)
                writer.write(image)
            writer.release()
            cap, fps, total = open_capture(file)
            try:
                self.assertEqual(total, 16)
                self.assertAlmostEqual(fps, 12., delta=.1)
                for requested in (0, 11, 3, 15):
                    frame, index = read_at(cap, requested, total)
                    self.assertIsNotNone(frame)
                    self.assertAlmostEqual(index, requested, delta=1)
                    self.assertAlmostEqual(float(frame[8, 8, 0]), 12 * requested, delta=12)
            finally:
                cap.release()

    def test_video_display_is_headless_renderable(self):
        atlas = SweepAtlas()
        frame = np.zeros((80, 120, 3), np.uint8)
        picture = video_display(frame, atlas, True, 25., "test.mp4",
                                5, 100, "Preview only")
        self.assertEqual(picture.shape, (628, 1520, 3))


if __name__ == "__main__":
    unittest.main()
