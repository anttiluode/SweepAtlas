#!/usr/bin/env python3
"""SweepAtlas video player: load, scrub, and stitch a video onto the atlas.

Run: python SweepAtlas.py --video myvideo.mp4
Or:  python SweepAtlasVideo.py        (opens a file chooser)

Seeking previews frames without modifying the mosaic. Resume with Space to
continue stitching, or press R to start a new atlas at the current position.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from SweepAtlas import SweepAtlas, build_display, label


WINDOW = "SweepAtlas  |  Video stitching"
BAR_LEFT, BAR_RIGHT = 24, 1496
BAR_Y0, BAR_Y1 = 574, 596


def choose_video():
    """Use the OS file dialog; return None when cancelled."""
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            chosen = filedialog.askopenfilename(
                parent=root, title="Open video for SweepAtlas",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                           ("All files", "*.*")])
        finally:
            root.destroy()
        return Path(chosen) if chosen else None
    except Exception as exc:
        print(f"Video file dialog unavailable: {exc}", file=sys.stderr)
        print("Use: python SweepAtlas.py --video path/to/video.mp4", file=sys.stderr)
        return None


def frame_from_x(x, total_frames):
    """Clamp timeline clicks and drags to a valid video frame."""
    if total_frames <= 1:
        return 0
    alpha = max(0., min(1., (x - BAR_LEFT) / (BAR_RIGHT - BAR_LEFT)))
    return int(round(alpha * (total_frames - 1)))


def format_time(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{(seconds // 60) % 60:02}:{seconds % 60:02}"


def open_capture(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise OSError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise OSError(f"OpenCV could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps < 1:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise OSError("Video does not report a usable frame count for seeking")
    return cap, fps, frame_count


def read_at(cap, frame_index, frame_count):
    """Seek and read one frame. The capture points to its successor."""
    frame_index = max(0, min(frame_count - 1, int(frame_index)))
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
        return None, frame_index
    ok, frame = cap.read()
    if not ok:
        return None, frame_index
    # Some codecs land on a neighboring frame. Use the reported position
    # when it is trustworthy; otherwise keep the requested index.
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if np.isfinite(pos) and 0 <= pos <= frame_count + 1:
        frame_index = max(0, min(frame_count - 1, int(round(pos)) - 1))
    return frame, frame_index


def video_display(frame, atlas, paused, fps, name, index, total, message):
    base = build_display(frame, atlas, paused, atlas.lost == 0, fps, "")
    canvas = np.full((628, 1520, 3), (25, 26, 30), np.uint8)
    canvas[:550, :] = base
    # Replace webcam-only labels/instructions.
    cv2.rectangle(canvas, (0, 0), (310, 43), (25, 26, 30), -1)
    label(canvas, "VIDEO SOURCE", (12, 30), (80, 220, 150), .67)
    cv2.rectangle(canvas, (760, 509), (1519, 549), (25, 26, 30), -1)
    label(canvas, "O open  SPACE play/pause  R reset atlas  S save  L relocalize  Q quit",
          (774, 529), (195, 200, 207), .42)
    if message:
        label(canvas, message[:95], (774, 508), (110, 240, 135), .49)
    cv2.rectangle(canvas, (0, 550), (1519, 627), (34, 36, 41), -1)
    label(canvas, f"{name[:90]}", (24, 569), (204, 210, 221), .45)
    cv2.rectangle(canvas, (BAR_LEFT, BAR_Y0), (BAR_RIGHT, BAR_Y1), (62, 67, 73), -1)
    ratio = index / max(1, total - 1)
    endpoint = int(round(BAR_LEFT + ratio * (BAR_RIGHT - BAR_LEFT)))
    cv2.rectangle(canvas, (BAR_LEFT, BAR_Y0), (endpoint, BAR_Y1), (78, 173, 123), -1)
    cv2.circle(canvas, (endpoint, (BAR_Y0 + BAR_Y1) // 2), 9, (120, 235, 169), -1, cv2.LINE_AA)
    label(canvas, f"{format_time(index / fps)} / {format_time((total - 1) / fps)}"
          f"   frame {index + 1}/{total}", (24, 615), (221, 224, 231), .47)
    label(canvas, "DRAG timeline  |  , / . one frame  |  J / K -/+ 5 sec  |  scrub pauses stitching",
          (555, 615), (190, 194, 206), .42)
    return canvas


def run_video(args, path=None):
    """Open a video player with a live atlas and interactive timeline."""
    source = Path(path) if path else choose_video()
    if source is None:
        return 0

    try:
        cap, fps, total = open_capture(source)
    except OSError as exc:
        print(exc, file=sys.stderr)
        return 2

    def fresh_atlas(predictor=True):
        return SweepAtlas(args.mode, args.match_width, args.min_inliers,
                          args.tile_size, args.max_tiles, predictor=predictor)

    atlas = fresh_atlas()
    frame, index = read_at(cap, 0, total)
    if frame is None:
        cap.release()
        print(f"Cannot decode initial video frame: {source}", file=sys.stderr)
        return 2

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1520, 628)
    paused = False
    scrubbing = False
    pending_seek = None
    force_relocalize = False
    message = "Video loaded; drag below to scrub (seeking never writes to atlas)"
    message_until = time.monotonic() + 7
    saved_once = False
    last_process_index = -1
    next_frame_at = time.monotonic() + 1. / fps

    def mouse(event, x, y, flags, param):
        nonlocal scrubbing, pending_seek, paused
        if total < 1:
            return
        if event == cv2.EVENT_LBUTTONDOWN and BAR_Y0 - 10 <= y <= BAR_Y1 + 10:
            scrubbing, paused = True, True
            pending_seek = frame_from_x(x, total)
        elif event == cv2.EVENT_MOUSEMOVE and scrubbing:
            pending_seek = frame_from_x(x, total)
        elif event == cv2.EVENT_LBUTTONUP and scrubbing:
            pending_seek = frame_from_x(x, total)
            scrubbing = False

    cv2.setMouseCallback(WINDOW, mouse)

    def save():
        nonlocal saved_once, message, message_until
        if atlas.tiles.bounds is None:
            message, message_until = "Nothing stitched yet", time.monotonic() + 4
            return
        dest = args.out / f"SweepAtlas_{source.stem}_{datetime.now():%Y%m%d_%H%M%S}.png"
        try:
            w, h, scale = atlas.tiles.save(dest, args.max_export_mp)
            print(f"SAVED {dest.resolve()} ({w}x{h}, scale {scale:.3f})", flush=True)
            message, message_until = f"Saved {dest.name} ({w}x{h})", time.monotonic() + 7
            saved_once = True
        except Exception as exc:
            print(f"Save failed: {exc}", file=sys.stderr)
            message, message_until = f"Save failed: {exc}", time.monotonic() + 7

    def jump(target):
        nonlocal frame, index, paused, force_relocalize, next_frame_at
        nonlocal message, message_until
        result, landed = read_at(cap, target, total)
        if result is None:
            message, message_until = "Seek failed; unsupported codec or frame", time.monotonic() + 4
            return
        frame, index = result, landed
        paused = True
        force_relocalize = True
        next_frame_at = time.monotonic() + 1. / fps
        message = "Preview only: Space resumes stitching; R starts a fresh atlas here"
        message_until = time.monotonic() + 6

    print(f"VIDEO {source}  ({total} frames, {fps:.2f} fps)")
    print("Timeline drag / , . frame step / J K 5 seconds / SPACE play / R reset / S save / O open")
    try:
        while True:
            now = time.monotonic()
            if pending_seek is not None:
                target = pending_seek
                pending_seek = None
                if target != index:
                    jump(target)

            if not paused and now >= next_frame_at:
                # The frame currently displayed has not been stitched yet:
                # process before advancing, but never process the same frame
                # twice when the user pauses/rewinds.
                if index != last_process_index:
                    try:
                        atlas.process(frame, relocalize=force_relocalize)
                    except (cv2.error, ValueError) as exc:
                        atlas.last_status = f"Alignment error: {str(exc)[:75]}"
                    last_process_index = index
                    force_relocalize = False
                ok, upcoming = cap.read()
                if ok:
                    frame = upcoming
                    reported = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if np.isfinite(reported) and 1 <= reported <= total + 1:
                        index = max(0, min(total - 1, int(round(reported)) - 1))
                    else:
                        index = min(index + 1, total - 1)
                    next_frame_at = time.monotonic() + max(1. / fps, 1. / args.process_hz)
                else:
                    paused = True
                    message, message_until = "End of video; scrub back, replay, or save", time.monotonic() + 8

            visible = message if time.monotonic() < message_until else ""
            display = video_display(frame, atlas, paused, fps, source.name, index, total, visible)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKeyEx(15)
            if key in (ord("q"), ord("Q"), 27):
                if not args.no_autosave and not saved_once:
                    save()
                break
            if key in (ord("s"), ord("S")):
                save()
            elif key == 32:
                paused = not paused
                next_frame_at = time.monotonic() + 1. / fps
            elif key in (ord("r"), ord("R")):
                atlas = fresh_atlas(atlas.predictor)
                last_process_index = -1
                saved_once = False
                force_relocalize = False
                message, message_until = "Fresh atlas at this video position", time.monotonic() + 5
            elif key in (ord("l"), ord("L")):
                force_relocalize = True
                message, message_until = "Will try older anchors on next frame", time.monotonic() + 4
            elif key in (ord("p"), ord("P")):
                atlas.predictor = not atlas.predictor
                message, message_until = f"Predictor: {atlas.predictor}", time.monotonic() + 3
            elif key in (ord(","), 2424832, 65361):
                jump(index - 1)
            elif key in (ord("."), 2555904, 65363):
                jump(index + 1)
            elif key in (ord("j"), ord("J")):
                jump(index - round(5 * fps))
            elif key in (ord("k"), ord("K")):
                jump(index + round(5 * fps))
            elif key in (ord("o"), ord("O")):
                chosen = choose_video()
                if chosen is not None:
                    try:
                        new_cap, new_fps, new_total = open_capture(chosen)
                        new_frame, new_index = read_at(new_cap, 0, new_total)
                        if new_frame is None:
                            raise OSError("Could not decode the first frame")
                    except OSError as exc:
                        if "new_cap" in locals():
                            new_cap.release()
                        message, message_until = str(exc), time.monotonic() + 7
                    else:
                        cap.release()
                        cap, fps, total = new_cap, new_fps, new_total
                        source, frame, index = chosen, new_frame, new_index
                        atlas = fresh_atlas(atlas.predictor)
                        saved_once, last_process_index, force_relocalize = False, -1, False
                        paused = False
                        next_frame_at = time.monotonic() + 1. / fps
                        message, message_until = f"Loaded {source.name}", time.monotonic() + 5
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
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


if __name__ == "__main__":
    from SweepAtlas import parse_args
    options = parse_args()
    raise SystemExit(run_video(options, options.video))
