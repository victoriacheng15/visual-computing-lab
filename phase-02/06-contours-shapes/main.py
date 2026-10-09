"""Module 06: Real-Time Contours, Shape Analysis & Geometry Lab.

Provides an interactive workbench for topological boundary extraction,
Douglas-Peucker polygon approximation, oriented bounding boxes, and geometric shape classification.
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
    ShapeConfig,
    ShapeMode,
    create_split_view,
    process_shapes,
)

from utils.camera import FPSMeter, ThreadedCamera  # noqa: E402


def nothing(_: int) -> None:
    """Empty callback required by cv2.createTrackbar."""
    pass


def setup_trackbars(window_name: str, config: ShapeConfig) -> None:
    """Create interactive threshold and approximation sliders."""
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Binary Thresh", window_name, config.binary_thresh, 255, nothing)
    cv2.createTrackbar("Epsilon %", window_name, config.epsilon_percent, 10, nothing)
    cv2.createTrackbar("Min Area / 100", window_name, int(config.min_area / 100), 50, nothing)


def reset_trackbars(window_name: str, config: ShapeConfig) -> None:
    """Reset OpenCV trackbar positions to configuration defaults."""
    cv2.setTrackbarPos("Binary Thresh", window_name, config.binary_thresh)
    cv2.setTrackbarPos("Epsilon %", window_name, config.epsilon_percent)
    cv2.setTrackbarPos("Min Area / 100", window_name, int(config.min_area / 100))


def read_trackbars(window_name: str, config: ShapeConfig) -> ShapeConfig:
    """Read current slider positions and return an updated ShapeConfig."""
    thresh = cv2.getTrackbarPos("Binary Thresh", window_name)
    eps = max(1, cv2.getTrackbarPos("Epsilon %", window_name))
    min_area_val = max(1, cv2.getTrackbarPos("Min Area / 100", window_name)) * 100.0

    return ShapeConfig(
        binary_thresh=thresh,
        epsilon_percent=eps,
        min_area=min_area_val,
        invert_binary=config.invert_binary,
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
    parser = argparse.ArgumentParser(description="Module 06: Contours, Shapes & Geometry")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    print("========================================")
    print(" Real-Time Contour & Shape Geometry Lab")
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
    config = ShapeConfig()
    current_mode = ShapeMode.CLASSIFIER

    window_name = "Module 06: Contours, Shapes & Geometry Lab"
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if has_display:
        setup_trackbars(window_name, config)

    print("\n[Controls]")
    print("  Key [1]: Geometric Shape Classifier (Triangles, Rectangles, Circles)")
    print("  Key [2]: Oriented Bounding Box (minAreaRect & orientation angle)")
    print("  Key [3]: Convex Hull & Defects (Enclosing convex polygon & valleys)")
    print("  Key [4]: Enclosing Circle & Ellipse fitting")
    print("  Key [5]: Contour Hierarchy Tree (Parent boundaries vs child holes)")
    print("  Key [I]: Invert binary threshold (dark vs light background)")
    print("  Sliders: Adjust threshold, Douglas-Peucker epsilon %, and min area")
    print("  Key [C]: Cycle camera device")
    print("  Key [S]: Save snapshot & dump detected shape metadata")
    print("  Key [Q]: Exit\n")

    try:
        frame_idx = 0
        while True:
            ret, frame, _ = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            fps = fps_meter.tick()

            # Always sync trackbars live when display is active
            if has_display:
                config = read_trackbars(window_name, config)

            t0 = time.perf_counter()
            overlay, mask, shapes = process_shapes(frame, current_mode, config)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            split_view = create_split_view(
                overlay,
                mask,
                left_title=f"Mode: {current_mode.value}",
                right_title="Segmentation Mask",
            )

            # Primary detected shape summary
            primary_label = shapes[0].name if shapes else "No Target"
            primary_angle = f"{shapes[0].orientation_deg:3.0f} deg" if shapes else "--"

            # HUD Line 1: Metrics
            perf_text = (
                f"FPS: {fps:5.1f} | Latency: {latency_ms:4.1f}ms | "
                f"Target: {primary_label} ({primary_angle}) | Count: {len(shapes)}"
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

            # HUD Line 2: Parameters
            inv_str = "INV" if config.invert_binary else "NORM"
            params_text = (
                f"Thresh: {config.binary_thresh:3d} ({inv_str}) | "
                f"Epsilon: {config.epsilon_percent}% | "
                f"Min Area: {int(config.min_area)}px"
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
                    ord("1"): (ShapeMode.CLASSIFIER, "Shape Classifier"),
                    ord("2"): (ShapeMode.ORIENTED_BBOX, "Oriented BBox"),
                    ord("3"): (ShapeMode.CONVEX_HULL, "Convex Hull & Defects"),
                    ord("4"): (ShapeMode.ENCLOSING_CIRCLE, "Enclosing Circle & Ellipse"),
                    ord("5"): (ShapeMode.HIERARCHY, "Contour Tree Hierarchy"),
                }

                match key:
                    case 27 | 113:  # ESC or 'q'
                        break

                    case k if k in mode_map:
                        current_mode, mode_name = mode_map[k]
                        config = ShapeConfig(invert_binary=config.invert_binary)
                        reset_trackbars(window_name, config)
                        print(f"[Mode] Activated: {mode_name} (Sliders reset)")

                    case 105:  # 'i' toggle invert
                        config.invert_binary = not config.invert_binary
                        state = "Inverted (Light on Dark)" if config.invert_binary else "Normal"
                        print(f"[Threshold] Inversion toggled: {state}")

                    case 115:  # 's' snapshot & JSON export
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        slug = current_mode.name.lower()
                        img_path = output_dir / f"shape_{slug}_{ts}.png"
                        json_path = output_dir / f"shape_{slug}_{ts}.json"

                        cv2.imwrite(str(img_path), split_view)
                        payload = {
                            "mode": current_mode.value,
                            "binary_thresh": config.binary_thresh,
                            "epsilon_percent": config.epsilon_percent,
                            "min_area": config.min_area,
                            "detected_count": len(shapes),
                            "shapes": [
                                {
                                    "name": s.name,
                                    "centroid": s.centroid,
                                    "area": s.area,
                                    "perimeter": s.perimeter,
                                    "circularity": round(s.circularity, 3),
                                    "vertex_count": s.vertex_count,
                                    "orientation_deg": round(s.orientation_deg, 1),
                                }
                                for s in shapes
                            ],
                            "timestamp": ts,
                        }
                        with open(json_path, "w") as f:
                            json.dump(payload, f, indent=2)
                        print(f"[Saved] Snapshot: {img_path}")
                        print(f"[Saved] Config:   {json_path}")

                    case 99:  # 'c' cycle camera device
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
