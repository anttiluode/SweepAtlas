"""Non-destructive contrast and sharpness controls for SweepAtlas.

Geometry, keyframe matching, and atlas tiles always retain the source pixels.
Only preview and exported PNGs receive these optional appearance adjustments.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

MAX_CONTRAST = 200
MAX_SHARPNESS = 200
HALO = 12


@lru_cache(maxsize=201)
def _contrast_lut(percent: int) -> np.ndarray:
    gain = percent / 100.0
    levels = np.arange(256, dtype=np.float32)
    return np.clip(np.rint((levels - 127.5) * gain + 127.5), 0, 255).astype(np.uint8)


@dataclass
class ImageAdjustments:
    """Contrast 100% and sharpness 0% reproduce the original pixels exactly."""

    contrast: int = 100
    sharpness: int = 0

    def __post_init__(self):
        self.contrast = int(np.clip(self.contrast, 0, MAX_CONTRAST))
        self.sharpness = int(np.clip(self.sharpness, 0, MAX_SHARPNESS))

    @property
    def active(self) -> bool:
        return self.contrast != 100 or self.sharpness != 0

    def attach(self, window: str) -> None:
        """Add two native OpenCV GUI trackbars beneath the current window."""
        cv2.createTrackbar("Contrast %", window, self.contrast, MAX_CONTRAST, lambda _: None)
        cv2.createTrackbar("Sharpness %", window, self.sharpness, MAX_SHARPNESS, lambda _: None)

    def read(self, window: str) -> None:
        """Read slider values; called from the UI loop before drawing/saving."""
        self.contrast = cv2.getTrackbarPos("Contrast %", window)
        self.sharpness = cv2.getTrackbarPos("Sharpness %", window)

    def reset(self, window: str) -> None:
        self.contrast, self.sharpness = 100, 0
        cv2.setTrackbarPos("Contrast %", window, self.contrast)
        cv2.setTrackbarPos("Sharpness %", window, self.sharpness)

    def apply(self, bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Enhance one BGR array; mask excludes unobserved/transparent regions."""
        if not self.active:
            return bgr
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Expected a uint8 BGR image")
        result = cv2.LUT(bgr, _contrast_lut(self.contrast)) if self.contrast != 100 else bgr
        if self.sharpness:
            # A 5x5 unsharp mask; mask-aware smoothing avoids artificial dark
            # halos at the boundary of unobserved atlas pixels.
            amount = self.sharpness / 100.0
            original = result.astype(np.float32)
            if mask is not None:
                valid = (mask != 0).astype(np.float32)
                smooth_rgb = cv2.GaussianBlur(original * valid[:, :, None],
                                              (5, 5), 1.05)
                smooth_alpha = cv2.GaussianBlur(valid, (5, 5), 1.05)
                smooth_rgb /= np.maximum(smooth_alpha[:, :, None], 1e-5)
            else:
                smooth_rgb = cv2.GaussianBlur(original, (5, 5), 1.05)
            result = np.clip(np.rint(original + amount * (original - smooth_rgb)),
                             0, 255).astype(np.uint8)
        if mask is not None:
            # Keep unobserved canvas/background pixels completely untouched.
            result = result.copy()
            result[mask == 0] = bgr[mask == 0]
        return result

    def apply_rgba_inplace(self, bgra: np.ndarray, stripe_height: int = 256) -> None:
        """Filter an exported BGRA image in horizontal stripes.

        One RGBA output image is already allocated by the export routine.
        Stripe halos preserve blur continuity without allocating a second
        full-resolution image; alpha remains untouched.
        """
        if not self.active:
            return
        if bgra.dtype != np.uint8 or bgra.ndim != 3 or bgra.shape[2] != 4:
            raise ValueError("Expected a uint8 BGRA export")
        if stripe_height <= HALO:
            raise ValueError("stripe_height must exceed halo size")
        height = bgra.shape[0]
        previous = None
        for y0 in range(0, height, stripe_height):
            y1 = min(height, y0 + stripe_height)
            current = bgra[y0:min(height, y1 + HALO)].copy()
            source = np.concatenate((previous, current), axis=0) if previous is not None else current
            top = len(previous) if previous is not None else 0
            processed = self.apply(source[:, :, :3], source[:, :, 3])
            # Capture original bottom-row context before replacing its RGB.
            count = min(HALO, y1-y0)
            previous = current[(y1-y0)-count:y1-y0].copy()
            bgra[y0:y1, :, :3] = processed[top:top + (y1-y0)]
