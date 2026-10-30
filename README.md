# SweepAtlas

**SweepAtlas** is a single-file Python webcam application that stitches overlapping camera views into a larger image as you move the camera.

![pic](pic.png)

The left side of the window shows the live webcam feed. The right side shows the growing mosaic. You can save the result as a PNG image.

## How it works

SweepAtlas uses OpenCV feature detection and matching to estimate how each new camera frame aligns with earlier views. It warps and blends captured pixels into a tiled image canvas, and keeps a simple camera-motion estimate to help choose previous views for matching. It can also try to re-anchor the current view to an older keyframe.

This is **classical computer-vision stitching**, not a trained AI model or a TransformerX implementation. It does not generate unseen scene content. Re-anchoring is not global SLAM optimization.

## Install and run

Requires Python 3.12 or 3.13 and a webcam.

```bat
python3.13 -m pip install opencv-python numpy
python3.13 SweepAtlas.py
```

For perspective-changing sweeps:

```bat
python3.13 SweepAtlas.py --mode homography
```

For a different webcam, use `--camera 1`. Saved PNGs go into `sweepatlas_output/` by default. Areas that were not observed are transparent.

## Controls

| Key | Action |
|---|---|
| **S** | Save the current mosaic |
| **Space** | Pause or resume stitching |
| **R** | Reset the mosaic |
| **L** | Search earlier views to recover alignment |
| **P** | Toggle the motion predictor |
| **Q / Esc** | Quit; auto-save if not already saved |

## Best results and limitations

Move the camera slowly, maintain good overlap, and scan a textured, mostly flat subject such as a poster, painting, or wall.

Sweeping around a 3D room, passing moving people, or changing perspective strongly can introduce ghosting and misalignment. The application does not perform depth reconstruction or global map correction.

To run a synthetic alignment check without a webcam:

```bat
python3.13 SweepAtlas.py --self-test
```
