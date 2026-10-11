# Visual Computing Exploration Lab

A progressive visual computing project on Linux (Fedora), progressing from 2D pixel-level digital signal processing to real-time 3D perception, graphics rendering, and cyber-physical digital twins.

---

## Roadmap Architecture

A progressive progression spanning 2D digital signal processing to real-time spatial computing:

- **Phase 1: 2D Foundations & Signal Processing**: Coordinate conventions, camera ingestion pipelines, convolutions, and spatial filtering.
- **Phase 2: Classical Computer Vision & Geometric Features**: Perceptual color segmentation, differential edge operators, contour topology, kinematic tracking, and local invariant features.
- **Phase 3: Machine Learning & Modern Perception**: Deep object detection, semantic segmentation, and neural perception architectures.
- **Phase 4: 3D Vision & Spatial Representation**: Multi-view stereo, epipolar geometry, depth estimation, point clouds, and visual SLAM.
- **Phase 5: Graphics, Rendering & Cyber-Physical Systems**: GPU rasterization, ray tracing, physics engines, and digital twin synchronization.

---

## Hardware Profile & Environment

- **Host OS:** Fedora Linux
- **Primary GPU:** NVIDIA GeForce RTX 4070 (8GB VRAM)
- **Sensors:** Standard USB / Integrated consumer webcams (V4L2)
- **GUI Backend:** OpenCV Qt with `QT_QPA_PLATFORM=xcb` (XWayland)
- **Package Manager:** `uv` with Python 3.12 pinned

---

## Getting Started

### Installation & Sync

Ensure `uv` is installed, then synchronize the environment:

```bash
# Sync core dependencies
make sync

# Run format and lint checks
make check
```

---

## Completed Modules Reference

| Module | Topic | Core Concepts |
| :---: | :--- | :--- |
| 01 | [Image Basics](phase-01/01-image-basics/README.md) | NumPy arrays, BGR vs RGB, slicing, memory contiguity |
| 02 | [Webcam & Real-Time Ingestion](phase-01/02-webcam/README.md) | Multi-threaded ingestion, zero-latency queue, synthetic fallback |
| 03 | [Spatial Image Processing](phase-01/03-image-processing/README.md) | Spatial convolutions, kernel scaling, bilateral and portrait blur |
| 04 | [Color Detection & HSV](phase-02/04-color-detection/README.md) | HSV thresholding, dual-interval red wrap, moment tracking |
| 05 | [Differential Edge Detection](phase-02/05-edge-detection/README.md) | Sobel, Scharr, Laplacian zero-crossings, Canny pipeline, angle map |
| 06 | [Contours & Shape Classification](phase-02/06-contours-shapes/README.md) | Suzuki contours, Douglas-Peucker, circularity, rotated boxes |
| 07 | [Object Tracking](phase-02/07-object-tracking/README.md) | Kalman filter kinematics, Lucas-Kanade, Farneback, CamShift |
| 08 | [Feature Detection & Matching](phase-02/08-feature-detection/README.md) | ORB, SIFT, FLANN matching, RANSAC homography, AR perspective warp |
| 09 | [Object Detection & NMS](phase-03/09-object-detection/README.md) | Scale pyramids, proposal storm, vectorized IoU, Non-Maximum Suppression |
| 10 | [Image Classification & Embeddings](phase-03/10-image-classification/README.md) | 128D visual embeddings, metric learning, cosine logits, Softmax, EMA smoothing |
| 11 | [Foreground Segmentation & Virtual BG](phase-03/11-segmentation/README.md) | MOG2/KNN statistical models, shadow suppression, alpha feathering, virtual compositing |

---

## Development & Code Quality

The repository enforces strict formatting and linting via Ruff:

```bash
# Verify lint rules and formatting
make check

# Auto-format all Python files
make format

# Auto-fix linting issues and re-format
make fix
```
