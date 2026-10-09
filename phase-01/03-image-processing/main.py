"""Module 03: Real-Time Spatial Image Processing & Filtering.

Processes live webcam stream through classical spatial filters:
box blur, Gaussian smoothing, high-pass sharpening, bilateral edge-preserving filter,
and alpha-blended portrait background blur.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repository root is on sys.path for shared utils
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# OpenCV bundled Qt on Linux only provides libqxcb (XWayland)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from filters import FILTER_NAMES, process_frame  # noqa: E402

from utils.camera import FPSMeter, ThreadedCamera  # noqa: E402


def detect_camera_indices() -> list[int]:
    """Scan and return device indices that stream video."""
    valid = []
    backend = cv2.CAP_V4L2 if os.name == "posix" else cv2.CAP_ANY
    for idx in range(8):
        dev_path = Path(f"/dev/video{idx}")
        if not dev_path.exists():
            continue
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                valid.append(idx)
    return valid if valid else [0]


def draw_hud(
    frame: np.ndarray,
    fps: float,
    filter_time_ms: float,
    mode_name: str,
    device_index: int,
    resolution: tuple[int, int],
) -> np.ndarray:
    """Render a clean telemetry HUD banner on top of the filtered video frame."""
    hud = frame.copy()
    w, h = resolution

    # Header banner
    overlay = hud.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.75, hud, 0.25, 0, hud)

    status_line = (
        f"FPS: {fps:5.1f} | Filter: {filter_time_ms:4.1f}ms | "
        f"Dev: {device_index} | Mode: {mode_name}"
    )
    cv2.putText(
        hud,
        status_line,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 240, 255),
        1,
        cv2.LINE_AA,
    )

    # Footer instructions
    cv2.putText(
        hud,
        "[1-7] Filter   [+/-] Blur Size   [C] Cam   [S] Snapshot   [Q] Quit",
        (12, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    return hud


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 03: Spatial Image Processing")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    print("========================================")
    print(" Visual Computing Lab - Module 03")
    print(" Real-Time Spatial Image Processing")
    print("========================================")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    available_cams = detect_camera_indices()
    current_device = args.device

    print(f"\n[Hardware] Available cameras: {available_cams}")
    print(f"[Hardware] Starting with camera index: {current_device}")

    cam = ThreadedCamera(
        device_index=current_device,
        width=1280,
        height=720,
        fps=30,
        use_synthetic=False,
    )
    cam.start()

    fps_meter = FPSMeter(window_size=30)
    current_filter_mode = 1
    current_ksize = 51
    window_name = "Module 03: Spatial Filtering & Convolution Lab"

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    print("\n[Controls]")
    for key, name in FILTER_NAMES.items():
        print(f"  Key [{key}]: {name}")
    print("  Keys [+/-]: Increase/decrease blur kernel size")
    print("  Key [C]: Cycle camera device")
    print("  Key [S]: Save snapshot")
    print("  Key [Q]: Exit\n")

    try:
        frame_idx = 0
        while True:
            ret, frame, _ = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            fps = fps_meter.tick()
            resolution = cam.get_resolution()

            # Measure filter execution time
            t0 = time.perf_counter()
            filtered_frame, mode_name = process_frame(
                frame, current_filter_mode, ksize=current_ksize
            )
            filter_time_ms = (time.perf_counter() - t0) * 1000.0

            annotated_frame = draw_hud(
                filtered_frame,
                fps,
                filter_time_ms,
                mode_name,
                cam.device_index,
                resolution,
            )

            if has_display:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                elif ord("1") <= key <= ord("7"):
                    current_filter_mode = key - ord("0")
                    print(f"[Filter] Selected: {FILTER_NAMES.get(current_filter_mode)}")
                elif key in (ord("+"), ord("=")):
                    current_ksize = min(151, current_ksize + 10)
                    print(f"[Kernel Size] Increased to {current_ksize}x{current_ksize}")
                elif key in (ord("-"), ord("_")):
                    current_ksize = max(3, current_ksize - 10)
                    print(f"[Kernel Size] Decreased to {current_ksize}x{current_ksize}")
                elif key == ord("s"):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = output_dir / f"filter_m{current_filter_mode}_{ts}.png"
                    cv2.imwrite(str(out_path), annotated_frame)
                    print(f"[Snapshot] Saved to: {out_path}")
                elif key == ord("c"):
                    if len(available_cams) > 1:
                        next_i = (available_cams.index(current_device) + 1) % len(available_cams)
                        current_device = available_cams[next_i]
                    else:
                        current_device = 2 if current_device == 0 else 0
                    print(f"\n[Switch] Reconnecting to camera {current_device}...")
                    cam.release()
                    cam = ThreadedCamera(
                        device_index=current_device,
                        width=1280,
                        height=720,
                        fps=30,
                        use_synthetic=False,
                    )
                    cam.start()
            else:
                frame_idx += 1
                if frame_idx % 10 == 0:
                    status = (
                        f"Frame {frame_idx:03d} | FPS: {fps:.1f} | "
                        f"Filter: {filter_time_ms:.2f}ms ({mode_name})"
                    )
                    print(f"[Headless] {status}")
                if frame_idx >= 30:
                    out_path = output_dir / "headless_filter_sample.png"
                    cv2.imwrite(str(out_path), annotated_frame)
                    print(f"[Headless] Sample saved to: {out_path}. Exiting.")
                    break

    finally:
        cam.release()
        if has_display:
            cv2.destroyAllWindows()
        print("\n[Shutdown] Filter pipeline released cleanly.")


if __name__ == "__main__":
    main()
