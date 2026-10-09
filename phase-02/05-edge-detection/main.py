"""Module 05: Real-Time Edge Detection & Spatial Gradients Lab.

Provides an interactive workbench comparing Sobel, Scharr, Laplacian,
Canny edge detection, and real-time HSV gradient angle vector field visualization.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repository root and module dir are importable
MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

for path in (MODULE_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# OpenCV bundled Qt on Linux only provides libqxcb (XWayland)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2  # noqa: E402
from detector import (  # noqa: E402
    EdgeConfig,
    EdgeMode,
    create_split_view,
    process_edges,
)

from utils.camera import FPSMeter, ThreadedCamera  # noqa: E402


def nothing(_: int) -> None:
    """Empty callback required by cv2.createTrackbar."""
    pass


def setup_trackbars(window_name: str, config: EdgeConfig) -> None:
    """Create interactive threshold and kernel sliders in the OpenCV window."""
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Canny Low", window_name, config.canny_lower, 255, nothing)
    cv2.createTrackbar("Canny High", window_name, config.canny_upper, 255, nothing)
    cv2.createTrackbar("Blur K (Odd)", window_name, config.blur_kernel, 15, nothing)


def reset_trackbars(window_name: str, config: EdgeConfig) -> None:
    """Reset OpenCV trackbar positions to configuration defaults."""
    cv2.setTrackbarPos("Canny Low", window_name, config.canny_lower)
    cv2.setTrackbarPos("Canny High", window_name, config.canny_upper)
    cv2.setTrackbarPos("Blur K (Odd)", window_name, config.blur_kernel)


def read_trackbars(window_name: str, config: EdgeConfig) -> EdgeConfig:
    """Read current slider values and return an updated configuration."""
    low = cv2.getTrackbarPos("Canny Low", window_name)
    high = cv2.getTrackbarPos("Canny High", window_name)
    blur_raw = cv2.getTrackbarPos("Blur K (Odd)", window_name)

    # Ensure blur kernel is an odd positive integer
    blur = max(1, blur_raw | 1)

    return EdgeConfig(
        canny_lower=low,
        canny_upper=high,
        blur_kernel=blur,
        sobel_kernel=config.sobel_kernel,
        laplacian_kernel=config.laplacian_kernel,
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 05: Edge Detection & Spatial Gradients")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    print("========================================")
    print(" Real-Time Edge & Gradient Laboratory")
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
    config = EdgeConfig()
    current_mode = EdgeMode.CANNY

    window_name = "Module 05: Edge & Gradient Exploration Lab"
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if has_display:
        setup_trackbars(window_name, config)

    print("\n[Controls]")
    print("  Key [1]: Canny Edge Pipeline (Single-pixel thin edges)")
    print("  Key [2]: Sobel Gradient Magnitude (First derivative)")
    print("  Key [3]: Scharr Gradient (Rotational symmetry)")
    print("  Key [4]: Laplacian (Second derivative zero-crossing)")
    print("  Key [5]: HSV Gradient Angle Map (Vector field orientation)")
    print("  Sliders: Adjust Canny Lower/Upper thresholds & Gaussian pre-blur")
    print("  Key [C]: Cycle camera device")
    print("  Key [S]: Save snapshot & dump JSON configuration")
    print("  Key [Q]: Exit\n")

    try:
        frame_idx = 0
        while True:
            ret, frame, _ = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            fps = fps_meter.tick()

            # Always sync trackbars live when display is present
            if has_display:
                config = read_trackbars(window_name, config)

            t0 = time.perf_counter()
            orig, edge_vis = process_edges(frame, current_mode, config)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            split_view = create_split_view(
                orig,
                edge_vis,
                left_title="Live Input Feed",
                right_title=f"Mode: {current_mode.value}",
            )

            # HUD telemetry overlay line 1: Performance metrics
            perf_text = (
                f"FPS: {fps:5.1f} | Latency: {latency_ms:4.1f}ms | Mode: {current_mode.value}"
            )
            cv2.putText(
                split_view,
                perf_text,
                (12, split_view.shape[0] - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 240, 255),
                1,
                cv2.LINE_AA,
            )

            # HUD telemetry overlay line 2: Current filter parameters
            if current_mode == EdgeMode.CANNY:
                params_text = (
                    f"Canny Low: {config.canny_lower:3d} | "
                    f"Canny High: {config.canny_upper:3d} | "
                    f"Blur Kernel: {config.blur_kernel}x{config.blur_kernel}"
                )
            else:
                params_text = (
                    f"Pre-blur Kernel: {config.blur_kernel}x{config.blur_kernel} | "
                    f"Sobel Aperture: {config.sobel_kernel}x{config.sobel_kernel}"
                )
            cv2.putText(
                split_view,
                params_text,
                (12, split_view.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (120, 255, 120),
                1,
                cv2.LINE_AA,
            )

            if has_display:
                cv2.imshow(window_name, split_view)
                key = cv2.waitKey(1) & 0xFF

                # Mode switch mapping for keys 1 - 5
                mode_map = {
                    ord("1"): (EdgeMode.CANNY, "Canny Edge Pipeline"),
                    ord("2"): (EdgeMode.SOBEL, "Sobel First Derivative"),
                    ord("3"): (EdgeMode.SCHARR, "Scharr Derivative"),
                    ord("4"): (EdgeMode.LAPLACIAN, "Laplacian Second Derivative"),
                    ord("5"): (EdgeMode.GRADIENT_ANGLE, "HSV Gradient Angle Vector Field"),
                }

                match key:
                    case 27 | 113:  # ESC or 'q'
                        break

                    case k if k in mode_map:
                        current_mode, mode_name = mode_map[k]
                        config = EdgeConfig()
                        reset_trackbars(window_name, config)
                        print(f"[Mode] Activated: {mode_name} (Sliders reset)")

                    case 115:  # 's'
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        slug = current_mode.name.lower()
                        img_path = output_dir / f"edge_{slug}_{ts}.png"
                        json_path = output_dir / f"edge_{slug}_{ts}.json"

                        cv2.imwrite(str(img_path), split_view)
                        payload = {
                            "mode": current_mode.value,
                            "canny_lower": config.canny_lower,
                            "canny_upper": config.canny_upper,
                            "blur_kernel": config.blur_kernel,
                            "sobel_kernel": config.sobel_kernel,
                            "laplacian_kernel": config.laplacian_kernel,
                            "timestamp": ts,
                        }
                        with open(json_path, "w") as f:
                            json.dump(payload, f, indent=2)
                        print(f"[Saved] Snapshot: {img_path}")
                        print(f"[Saved] Config:   {json_path}")

                    case 99:  # 'c'
                        if len(available_cams) > 1:
                            cam.release()
                        curr_idx = available_cams.index(current_device)
                        next_idx = (curr_idx + 1) % len(available_cams)
                        current_device = available_cams[next_idx]
                        print(f"\n[Hardware] Switching to camera: {current_device}")
                        cam = ThreadedCamera(
                            device_index=current_device,
                            width=1280,
                            height=720,
                            fps=30,
                            use_synthetic=False,
                        )
                        cam.start()

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[App] Interrupted by user.")
    finally:
        cam.release()
        if has_display:
            cv2.destroyAllWindows()
        print("[App] Shutdown cleanly.")


if __name__ == "__main__":
    main()
