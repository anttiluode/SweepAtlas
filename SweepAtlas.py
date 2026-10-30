#!/usr/bin/env python3
"""SweepAtlas — real webcam-to-mosaic stitcher, in one Python file.

Install:  py -3.13 -m pip install opencv-python numpy
Run:      py -3.13 SweepAtlas.py
Test:     py -3.13 SweepAtlas.py --self-test

A classical, verified visual-geometry implementation, not a trained TransformerX
model. Its persistent pose/velocity state guides matching; source pixels alone
supply the image. Intended for textured planar scenes and careful panorama sweeps.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def transform_points(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2),
                                    np.asarray(h, np.float64)).reshape(-1, 2)


def center_of(h: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, ht = size
    return transform_points(h, [(w * .5, ht * .5)])[0]


@dataclass
class Keyframe:
    number: int
    keypoints: list
    descriptors: np.ndarray | None
    pose: np.ndarray                       # local full-resolution pixels -> atlas pixels
    center: np.ndarray


@dataclass
class Match:
    pose: np.ndarray
    inliers: int
    good: int
    error: float
    reference: Keyframe


class AtlasTiles:
    """Unbounded logical coordinates, with physical tiles allocated on first use."""

    def __init__(self, tile_size: int = 512, max_tiles: int = 1250):
        self.ts = tile_size
        self.max_tiles = max_tiles
        self.tiles: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.bounds: tuple[int, int, int, int] | None = None
        self.feather = None
        self.feather_shape = None

    def _weights(self, height: int, width: int) -> np.ndarray:
        if self.feather_shape != (height, width):
            margin = max(16, int(min(height, width) * .14))
            x = np.minimum(np.arange(width) + 1, np.arange(width, 0, -1))
            y = np.minimum(np.arange(height) + 1, np.arange(height, 0, -1))
            wx = np.clip(x.astype(np.float32) / margin, .02, 1)
            wy = np.clip(y.astype(np.float32) / margin, .02, 1)
            self.feather = np.minimum(wy[:, None], wx[None, :]).astype(np.float32)
            self.feather_shape = (height, width)
        return self.feather

    def add(self, frame: np.ndarray, pose: np.ndarray) -> bool:
        h, w = frame.shape[:2]
        corners = transform_points(pose, [(0, 0), (w, 0), (w, h), (0, h)])
        if not np.isfinite(corners).all():
            return False
        x0, y0 = np.floor(corners.min(axis=0)).astype(int) - 2
        x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + 2
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0 or bw > 2.8 * w or bh > 2.8 * h:
            return False
        if bw * bh > 10_000_000:
            return False
        ts = self.ts
        needed = {(tx, ty) for tx in range(x0 // ts, (x1 - 1) // ts + 1)
                              for ty in range(y0 // ts, (y1 - 1) // ts + 1)}
        if len(self.tiles.keys() | needed) > self.max_tiles:
            return False
        relative = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64) @ pose
        warped = cv2.warpPerspective(frame, relative, (bw, bh), flags=cv2.INTER_LINEAR)
        weight = cv2.warpPerspective(self._weights(h, w), relative, (bw, bh),
                                     flags=cv2.INTER_LINEAR)
        valid_rect = weight > .006
        if not np.any(valid_rect):
            return False
        for tx, ty in needed:
            tile_x, tile_y = tx * ts, ty * ts
            left, top = max(x0, tile_x), max(y0, tile_y)
            right, bottom = min(x1, tile_x + ts), min(y1, tile_y + ts)
            if left >= right or top >= bottom:
                continue
            source = (slice(top - y0, bottom - y0), slice(left - x0, right - x0))
            incoming_weight = weight[source]
            if not np.any(valid_rect[source]):
                continue
            if (tx, ty) not in self.tiles:
                self.tiles[(tx, ty)] = (np.zeros((ts, ts, 3), dtype=np.uint8),
                                        np.zeros((ts, ts), dtype=np.float32))
            rgb_tile, w_tile = self.tiles[(tx, ty)]
            dest = (slice(top - tile_y, bottom - tile_y),
                    slice(left - tile_x, right - tile_x))
            old_rgb = rgb_tile[dest]
            old_weight = w_tile[dest]
            # Weighted average; cap historic weight so later confirmed views can
            # still correct alignment/exposure. No synthesized image pixels.
            new_weight = np.where(valid_rect[source], incoming_weight, 0)
            mix = (new_weight / (old_weight + new_weight + 1e-9))[:, :, None]
            blended = old_rgb.astype(np.float32) * (1 - mix) + warped[source].astype(np.float32) * mix
            old_rgb[:] = np.clip(blended, 0, 255).astype(np.uint8)
            old_weight[:] = np.minimum(old_weight + new_weight, 6.0)
        if self.bounds is None:
            self.bounds = (x0, y0, x1, y1)
        else:
            a, b, c, d = self.bounds
            self.bounds = (min(a, x0), min(b, y0), max(c, x1), max(d, y1))
        return True

    def preview(self, width: int, height: int, camera_pose: np.ndarray | None,
                frame_size: tuple[int, int] | None, keyframes: list[Keyframe]) -> np.ndarray:
        view = np.full((height, width, 3), (29, 31, 35), np.uint8)
        if self.bounds is None:
            return view
        x0, y0, x1, y1 = self.bounds
        extent_w, extent_h = max(1, x1-x0), max(1, y1-y0)
        scale = min((width - 24) / extent_w, (height - 24) / extent_h)
        ox = (width - extent_w * scale) * .5
        oy = (height - extent_h * scale) * .5
        ts = self.ts
        for (tx, ty), (rgb, weights) in self.tiles.items():
            u0 = int(round((tx * ts - x0) * scale + ox))
            v0 = int(round((ty * ts - y0) * scale + oy))
            u1 = int(round(((tx+1)*ts - x0) * scale + ox))
            v1 = int(round(((ty+1)*ts - y0) * scale + oy))
            if u1 <= u0 or v1 <= v0:
                continue
            l, t, r, b = max(0, u0), max(0, v0), min(width, u1), min(height, v1)
            if r <= l or b <= t:
                continue
            patch_w, patch_h = u1-u0, v1-v0
            resized = cv2.resize(rgb, (patch_w, patch_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            mask = cv2.resize((weights > .006).astype(np.uint8), (patch_w, patch_h),
                              interpolation=cv2.INTER_NEAREST)
            ry, rx = slice(t-v0, b-v0), slice(l-u0, r-u0)
            dest = view[t:b, l:r]
            np.copyto(dest, resized[ry, rx], where=mask[ry, rx, None].astype(bool))
        for key in keyframes[-120:]:
            px = int((key.center[0] - x0) * scale + ox)
            py = int((key.center[1] - y0) * scale + oy)
            if 0 <= px < width and 0 <= py < height:
                cv2.circle(view, (px, py), 2, (190, 130, 65), -1, cv2.LINE_AA)
        if camera_pose is not None and frame_size is not None:
            w, h = frame_size
            quad = transform_points(camera_pose, [(0, 0), (w, 0), (w, h), (0, h)])
            quad = np.rint((quad - [x0, y0]) * scale + [ox, oy]).astype(np.int32)
            cv2.polylines(view, [quad], True, (60, 220, 130), 2, cv2.LINE_AA)
        return view

    def save(self, path: Path, max_megapixels: float = 100.0) -> tuple[int, int, float]:
        if self.bounds is None:
            raise RuntimeError('No mosaic content yet')
        x0, y0, x1, y1 = self.bounds
        width, height = x1 - x0, y1 - y0
        factor = min(1., math.sqrt(max_megapixels * 1_000_000 / max(1, width * height)))
        export_w = max(1, int(math.ceil(width * factor)))
        export_h = max(1, int(math.ceil(height * factor)))
        # BGRA: areas never observed are transparent, not invented content.
        out = np.zeros((export_h, export_w, 4), dtype=np.uint8)
        ts = self.ts
        for (tx, ty), (rgb, weights) in self.tiles.items():
            left, top = tx * ts, ty * ts
            right, bottom = left + ts, top + ts
            l, t = max(0, int((left-x0)*factor)), max(0, int((top-y0)*factor))
            r, b = min(export_w, int(math.ceil((right-x0)*factor))), min(export_h, int(math.ceil((bottom-y0)*factor)))
            if r <= l or b <= t:
                continue
            src_x = (np.arange(l, r) + .5) / factor + x0 - left
            src_y = (np.arange(t, b) + .5) / factor + y0 - top
            indices_x = np.clip(src_x.astype(np.int32), 0, ts-1)
            indices_y = np.clip(src_y.astype(np.int32), 0, ts-1)
            patch = rgb[np.ix_(indices_y, indices_x)]
            alpha = ((weights[np.ix_(indices_y, indices_x)] > .006) * 255).astype(np.uint8)
            # Every output pixel's tile is unique except possible roundoff at
            # scale reductions; the tile-grid partition remains deterministic.
            out[t:b, l:r, :3] = patch
            out[t:b, l:r, 3] = alpha
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise OSError(f'Cannot write {path}')
        return export_w, export_h, factor


class SweepAtlas:
    def __init__(self, mode='affine', work_width=640, min_inliers=18, tile_size=512,
                 max_tiles=1250, predictor=True):
        self.mode = mode
        self.work_width = work_width
        self.min_inliers = min_inliers
        self.tiles = AtlasTiles(tile_size, max_tiles)
        self.orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8,
                                  edgeThreshold=18, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.keyframes: list[Keyframe] = []
        self.pose: np.ndarray | None = None
        self.size: tuple[int, int] | None = None
        self.number = 0
        self.added = 0
        self.lost = 0
        self.closures = 0
        self.velocity = np.zeros(2, np.float64)
        self.predictor = predictor
        self.last_inliers = 0
        self.last_status = 'Move slowly across a textured, mostly flat scene.'
        self.last_add_center: np.ndarray | None = None
        self.last_added_at = 0.

    def _features(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        if self.size is not None and self.size != (w, h):
            raise ValueError('Camera resolution changed during capture')
        factor = min(1., self.work_width / w)
        work = cv2.resize(frame, (round(w*factor), round(h*factor)),
                          interpolation=cv2.INTER_AREA) if factor < 1 else frame
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        kps, descriptors = self.orb.detectAndCompute(gray, None)
        return kps, descriptors, factor

    def _match(self, keypoints, desc, factor, reference: Keyframe):
        if desc is None or reference.descriptors is None:
            return None
        if len(desc) < 16 or len(reference.descriptors) < 16:
            return None
        pairs = self.matcher.knnMatch(desc, reference.descriptors, k=2)
        good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < .76 * b.distance]
        good.sort(key=lambda m: m.distance)
        used = set()
        unique = []
        for m in good:
            if m.trainIdx not in used:
                unique.append(m)
                used.add(m.trainIdx)
        if len(unique) < self.min_inliers:
            return None
        src = np.float32([keypoints[m.queryIdx].pt for m in unique])
        dst = np.float32([reference.keypoints[m.trainIdx].pt for m in unique])
        if self.mode == 'homography':
            raw, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.8,
                                           maxIters=1800, confidence=.995)
        else:
            affine, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                       ransacReprojThreshold=3.5,
                                                       maxIters=1800,
                                                       confidence=.995, refineIters=10)
            raw = np.vstack([affine, [0, 0, 1]]) if affine is not None else None
        if raw is None or mask is None:
            return None
        good_mask = mask.reshape(-1).astype(bool)
        n = int(good_mask.sum())
        if n < self.min_inliers or n / len(unique) < .38:
            return None
        dim = np.array([[1 / factor, 0, 0], [0, 1 / factor, 0], [0, 0, 1]], np.float64)
        relative = dim @ raw @ np.linalg.inv(dim)
        full_w, full_h = self.size
        corners = transform_points(relative, [(0, 0), (full_w, 0), (full_w, full_h), (0, full_h)])
        if not np.isfinite(corners).all():
            return None
        area = abs(float(cv2.contourArea(corners.astype(np.float32))))
        if not (.50 * full_w * full_h <= area <= 2.0 * full_w * full_h):
            return None
        if np.max(np.abs(corners)) > 5 * max(full_w, full_h):
            return None
        error = float(np.median(np.linalg.norm(transform_points(raw, src[good_mask]) - dst[good_mask], axis=1)))
        if error > 3.8:
            return None
        # Matches must cover a meaningful area, not an accidental line or corner.
        support = np.ptp(src[good_mask], axis=0)
        if support[0] < .10 * full_w * factor or support[1] < .09 * full_h * factor:
            return None
        candidate_pose = reference.pose @ relative
        if abs(candidate_pose[2, 2]) < 1e-10:
            return None
        candidate_pose = candidate_pose / candidate_pose[2, 2]
        c = center_of(candidate_pose, (full_w, full_h))
        if not np.isfinite(c).all() or np.max(np.abs(c)) > 100_000:
            return None
        return Match(candidate_pose, n, len(unique), error, reference)

    def _candidate_references(self, force_relocalize=False):
        if not self.keyframes:
            return []
        if force_relocalize:
            # Older keyframes help re-anchor an atlas when the camera revisits
            # an already observed area after losing its immediate predecessor.
            return self.keyframes[-140:]
        if self.pose is None:
            return self.keyframes[-1:]
        prediction = center_of(self.pose, self.size) + (self.velocity if self.predictor else 0)
        latest = self.keyframes[-1]
        choices = [latest]
        others = sorted(self.keyframes[:-1], key=lambda k: np.linalg.norm(k.center - prediction))
        diagonal = float(np.hypot(*self.size))
        for k in others:
            if len(choices) >= 6:
                break
            if np.linalg.norm(k.center - prediction) < diagonal * 1.3:
                choices.append(k)
        return choices

    def process(self, frame: np.ndarray, relocalize=False) -> bool:
        if frame is None or frame.ndim != 3:
            self.last_status = 'Bad camera frame'
            return False
        height, width = frame.shape[:2]
        if self.size is None:
            self.size = (width, height)
        self.number += 1
        kps, desc, factor = self._features(frame)
        if self.pose is None:
            self.pose = np.eye(3, dtype=np.float64)
            if not self.tiles.add(frame, self.pose):
                self.last_status = 'Cannot allocate the first atlas tile'
                return False
            c = center_of(self.pose, self.size)
            self.keyframes.append(Keyframe(self.number, kps, desc, self.pose.copy(), c))
            self.last_add_center = c
            self.last_added_at = time.monotonic()
            self.added += 1
            self.last_status = f'Atlas initialized ({len(kps)} camera features)'
            return True
        refs = self._candidate_references(relocalize or self.lost >= 4)
        matches = [m for ref in refs if (m := self._match(kps, desc, factor, ref)) is not None]
        if not matches:
            self.lost += 1
            self.last_inliers = 0
            self.last_status = f'Alignment lost ({self.lost}); slow down / revisit textured area'
            return False
        predicted = center_of(self.pose, self.size) + (self.velocity if self.predictor else 0)
        # Inlier count dominates. Prefer geometrically reasonable motion and
        # lower residual among otherwise similarly strong candidates.
        def rank(m: Match):
            dist = np.linalg.norm(center_of(m.pose, self.size) - predicted)
            return m.inliers - .45 * m.error - .004 * dist
        best = max(matches, key=rank)
        old_center = center_of(self.pose, self.size)
        new_center = center_of(best.pose, self.size)
        shift = float(np.linalg.norm(new_center - old_center))
        # Reject implausible teleportation after a failed match / repeated motif.
        limit = (max(width, height) * (2.0 if self.lost >= 4 else 1.05))
        if shift > limit:
            self.lost += 1
            self.last_status = 'Rejected implausible camera jump'
            return False
        self.pose = best.pose
        self.velocity = .55 * self.velocity + .45 * (new_center - old_center)
        self.lost = 0
        self.last_inliers = best.inliers
        closure = self.number - best.reference.number >= 22
        if closure and shift > 3:
            self.closures += 1
        # Only blend when genuinely viewing a new region or after a pause.
        already = float(np.linalg.norm(new_center - self.last_add_center)) if self.last_add_center is not None else 1e9
        now = time.monotonic()
        should_add = already >= 13 or now - self.last_added_at > 2.5
        if should_add:
            ok = self.tiles.add(frame, self.pose)
            if not ok:
                self.last_status = 'Atlas tile limit or invalid projection; save mosaic now'
                return False
            self.added += 1
            self.last_add_center = new_center.copy()
            self.last_added_at = now
        # A new keyframe provides a stable atlas anchor for later matching.
        latest = self.keyframes[-1]
        distance = np.linalg.norm(new_center - latest.center)
        if ((distance >= min(width, height) * .27 or self.number - latest.number >= 24)
                and len(kps) >= self.min_inliers):
            self.keyframes.append(Keyframe(self.number, kps, desc, self.pose.copy(), new_center.copy()))
            if len(self.keyframes) > 240:
                # Keep anchors spread over the sweep plus all recent frames.
                self.keyframes = self.keyframes[::2][:-48] + self.keyframes[-48:]
        tag = ' | re-anchored to earlier view' if closure else ''
        self.last_status = f'{best.inliers}/{best.good} inliers; {best.error:.1f}px residual; {shift:.0f}px motion{tag}'
        return True


def label(image: np.ndarray, text: str, position: tuple[int, int], color=(238,238,238), size=.54):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def build_display(frame: np.ndarray, atlas: SweepAtlas, paused: bool, running: bool,
                  source_fps: float, message: str = ''):
    left_w, left_h = 760, 440
    right_w, right_h = 760, 440
    left = cv2.resize(frame, (left_w, left_h), interpolation=cv2.INTER_AREA)
    right = atlas.tiles.preview(right_w, right_h, atlas.pose, atlas.size, atlas.keyframes)
    image = np.full((550, left_w + right_w, 3), (25, 26, 30), np.uint8)
    image[45:485, :left_w] = left
    image[45:485, left_w:] = right
    cv2.rectangle(image, (left_w-2, 0), (left_w+1, 485), (70, 75, 85), -1)
    label(image, 'LIVE CAMERA', (12, 30), (80, 220, 150), .67)
    label(image, 'OBSERVED PIXEL ATLAS', (left_w + 12, 30), (80, 220, 150), .67)
    status = ('PAUSED' if paused else 'TRACKING' if running else 'WAITING')
    label(image, f'{status}  |  frames stitched: {atlas.added}  |  tiles: {len(atlas.tiles.tiles)}'
                 f'  |  anchors: {len(atlas.keyframes)}  |  re-anchors: {atlas.closures}',
          (14, 507), (205, 215, 230), .52)
    label(image, atlas.last_status[:118], (14, 529), (115, 192, 235), .50)
    if message:
        label(image, message[:95], (left_w + 12, 507), (110, 240, 135), .52)
    label(image, f'{source_fps:.1f} camera fps  |  O open video  S save  R reset  SPACE pause  L relocalize  P predictor  Q quit',
          (left_w + 12, 529), (195, 200, 207), .37)
    return image


def demo_self_test() -> None:
    rng = np.random.default_rng(20261030)
    scene = rng.integers(10, 245, (950, 1750, 3), dtype=np.uint8)
    scene = cv2.GaussianBlur(scene, (3, 3), .5)
    for i in range(120):
        center = int(rng.integers(5, 1745)), int(rng.integers(5, 945))
        color = tuple(int(x) for x in rng.integers(10, 255, size=3))
        cv2.circle(scene, center, int(rng.integers(4, 25)), color, -1, cv2.LINE_AA)
    atlas = SweepAtlas(mode='affine', work_width=480, min_inliers=14, tile_size=256)
    positions = [(110 + 45*i, 240 + (i//4)*12) for i in range(19)]
    positions += list(reversed(positions[:-1]))
    outcomes=[]
    for x,y in positions:
        frame = scene[y:y+360, x:x+640]
        outcomes.append(atlas.process(frame))
    assert sum(outcomes) >= int(.70 * len(outcomes)), f'too many tracking failures: {sum(outcomes)}/{len(outcomes)}'
    assert atlas.tiles.bounds is not None
    x0,y0,x1,y1 = atlas.tiles.bounds
    assert x1-x0 > 1050, f'not enough scene covered: {x1-x0}'
    assert len(atlas.tiles.tiles) >= 4
    assert atlas.last_inliers >= 14
    print(f'SELF-TEST PASS: {sum(outcomes)}/{len(outcomes)} frames tracked, '
          f'{atlas.added} blended, atlas {x1-x0}x{y1-y0}, '
          f'{atlas.closures} revisits, {len(atlas.tiles.tiles)} tiles')


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Real webcam sweep-to-image atlas: OpenCV + geometric verification')
    ap.add_argument('--video', type=Path, default=None,
                    help='Open a video with a scrubbable timeline instead of the webcam')
    ap.add_argument('--camera', type=int, default=0, help='Webcam index (try 1 for a second camera)')
    ap.add_argument('--width', type=int, default=960, help='Requested camera frame width')
    ap.add_argument('--height', type=int, default=540, help='Requested camera frame height')
    ap.add_argument('--process-hz', type=float, default=5., help='Feature matching attempts per second')
    ap.add_argument('--mode', choices=['affine','homography'], default='affine',
                    help='affine for near-frontoparallel flat scans; homography for perspective/panorama')
    ap.add_argument('--match-width', type=int, default=640, help='ORB working width (lower is faster)')
    ap.add_argument('--min-inliers', type=int, default=18)
    ap.add_argument('--tile-size', type=int, default=512)
    ap.add_argument('--max-tiles', type=int, default=1250)
    ap.add_argument('--max-export-mp', type=float, default=100.,
                    help='Downscale saved PNG only above this many megapixels')
    ap.add_argument('--out', type=Path, default=Path('sweepatlas_output'))
    ap.add_argument('--no-autosave', action='store_true', help='Do not save automatically on exit')
    ap.add_argument('--self-test', action='store_true', help='Synthetic geometric test, no webcam')
    args = ap.parse_args(argv)
    if (min(args.width,args.height,args.match_width,args.tile_size,args.min_inliers,args.max_tiles) < 1
        or args.process_hz <= 0 or args.max_export_mp <= 0):
        ap.error('Dimensions, processing frequency and limits must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        demo_self_test()
        return 0
    if args.video is not None:
        from SweepAtlasVideo import run_video
        return run_video(args, args.video)
    print('SweepAtlas — REAL webcam mosaic (no synthetic tiles, no hallucinated pixels)')
    print('Best results: textured wall/art/document; move slowly with good overlap and steady lighting.')
    print('Keys: O open video, S save, R reset, SPACE pause, L relocalize, P predictor, Q/Esc save+quit.')
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
    if not cap.isOpened() and sys.platform == 'win32':
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'Cannot open camera {args.camera}. Try --camera 1 or close Teams/Zoom/Camera.', file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    atlas = SweepAtlas(args.mode, args.match_width, args.min_inliers,
                       args.tile_size, args.max_tiles)
    title = 'SweepAtlas  |  Real webcam stitching'
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1520, 550)
    paused = False
    relocalize = False
    last_process = 0.
    last_frame = None
    last_overlay = 0.
    message, message_until = '', 0.
    count = 0
    last_fps_t = time.monotonic()
    fps = 0.
    saved_once = False

    def save():
        nonlocal saved_once, message, message_until
        if atlas.tiles.bounds is None:
            message, message_until = 'No image to save', time.monotonic()+4
            return
        filename = args.out / f'SweepAtlas_{datetime.now():%Y%m%d_%H%M%S}.png'
        print(f'Saving {filename} ...', flush=True)
        try:
            width, height, factor = atlas.tiles.save(filename, args.max_export_mp)
            print(f'SAVED {filename.resolve()}  ({width}x{height}, export scale {factor:.3f})', flush=True)
            message, message_until = f'Saved {filename.name} ({width}x{height})', time.monotonic()+8
            saved_once = True
        except Exception as exc:
            print('Save failed:', exc, file=sys.stderr)
            message, message_until = f'Save failed: {exc}', time.monotonic()+8

    try:
        while True:
            okay, frame = cap.read()
            if not okay:
                if last_frame is None:
                    print('Webcam returned no image.', file=sys.stderr)
                    return 2
                frame = last_frame.copy()
            else:
                last_frame = frame
            now = time.monotonic()
            count += 1
            if now - last_fps_t >= 1.0:
                fps = count / (now-last_fps_t)
                count, last_fps_t = 0, now
            if not paused and now-last_process >= 1.0/args.process_hz:
                try:
                    atlas.process(frame, relocalize=relocalize)
                except cv2.error as exc:
                    atlas.last_status = f'OpenCV alignment error: {str(exc)[:80]}'
                relocalize = False
                last_process = now
            visible_message = message if now < message_until else ''
            if now - last_overlay >= .08:
                display = build_display(frame, atlas, paused, atlas.lost==0, fps, visible_message)
                cv2.imshow(title, display)
                last_overlay = now
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), ord('Q'), 27):
                if not args.no_autosave and not saved_once:
                    save()
                break
            if key in (ord('s'), ord('S')):
                save()
            elif key in (ord('r'), ord('R')):
                atlas = SweepAtlas(args.mode, args.match_width, args.min_inliers,
                                   args.tile_size, args.max_tiles, predictor=atlas.predictor)
                saved_once = False
                message, message_until = 'Atlas reset', now+4
            elif key == 32:
                paused = not paused
            elif key in (ord('l'), ord('L')):
                relocalize = True
                message, message_until = 'Searching earlier keyframes', now+3
            elif key in (ord('p'), ord('P')):
                atlas.predictor = not atlas.predictor
                message, message_until = f'Pose predictor: {atlas.predictor}', now+3
            elif key in (ord('o'), ord('O')):
                from SweepAtlasVideo import choose_video, run_video
                video_path = choose_video()
                if video_path is not None:
                    cap.release()
                    cv2.destroyAllWindows()
                    return run_video(args, video_path)
            try:
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    if not args.no_autosave and not saved_once:
                        save()
                    break
            except cv2.error:
                break
    except KeyboardInterrupt:
        if not args.no_autosave and not saved_once:
            save()
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
