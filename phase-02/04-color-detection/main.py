"""Module 04: Real-Time Color Detection & HSV Calibration.

Tracks physical objects by color in HSV space, handles red hue wrap-around,
and provides an interactive calibration tool with live trackbars and split-screen telemetry.
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
    COLOR_PRESETS,
    HSVRange,
    annotate_frame,
    build_color_mask,
    create_split_view,
    extract_target_geometry,
)

from utils.camera import FPSMeter, ThreadedCamera  # noqa: E402


def nothing(_: int) -> None:
    """Empty callback required by cv2.createTrackbar."""
    pass


def setup_trackbars(window_name: str, initial: HSVRange) -> None:
    """Create interactive HSV calibration sliders in an OpenCV window."""
    cv2.namedWindow(window_name)
    cv2.createTrackbar("H Min", window_name, initial.h_min, 179, nothing)
    cv2.createTrackbar("H Max", window_name, initial.h_max, 179, nothing)
    cv2.createTrackbar("S Min", window_name, initial.s_min, 255, nothing)
    cv2.createTrackbar("S Max", window_name, initial.s_max, 255, nothing)
    cv2.createTrackbar("V Min", window_name, initial.v_min, 255, nothing)
    cv2.createTrackbar("V Max", window_name, initial.v_max, 255, nothing)


def read_trackbars(window_name: str) -> HSVRange:
    """Read current slider values from OpenCV trackbars."""
    h_min = cv2.getTrackbarPos("H Min", window_name)
    h_max = cv2.getTrackbarPos("H Max", window_name)
    s_min = cv2.getTrackbarPos("S Min", window_name)
    s_max = cv2.getTrackbarPos("S Max", window_name)
    v_min = cv2.getTrackbarPos("V Min", window_name)
    v_max = cv2.getTrackbarPos("V Max", window_name)

    return HSVRange(
        name="Custom (Trackbars)",
        h_min=h_min,
        h_max=h_max,
        s_min=s_min,
        s_max=s_max,
        v_min=v_min,
        v_max=v_max,
        wrap_red=(h_min > h_max),
    )


def update_trackbars(window_name: str, hsv: HSVRange) -> None:
    """Synchronize trackbar slider positions to a preset."""
    cv2.setTrackbarPos("H Min", window_name, hsv.h_min)
    cv2.setTrackbarPos("H Max", window_name, hsv.h_max)
    cv2.setTrackbarPos("S Min", window_name, hsv.s_min)
    cv2.setTrackbarPos("S Max", window_name, hsv.s_max)
    cv2.setTrackbarPos("V Min", window_name, hsv.v_min)
    cv2.setTrackbarPos("V Max", window_name, hsv.v_max)


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
    parser = argparse.ArgumentParser(description="Module 04: Color Detection & HSV Calibration")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    print("========================================")
    print(" Visual Computing Lab - Module 04")
    print(" Real-Time Color Segmentation & Tracking")
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
    active_range = COLOR_PRESETS[1]

    window_name = "Module 04: Color Detection & Calibration Lab"
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if has_display:
        setup_trackbars(window_name, active_range)

    print("\n[Controls]")
    print("  Key [1]: Red preset (Hue wrap-around)")
    print("  Key [2]: Green preset")
    print("  Key [3]: Blue preset")
    print("  Key [4]: Yellow preset")
    print("  Sliders: Drag H/S/V trackbars to adjust range live")
    print("  Key [C]: Cycle camera device")
    print("  Key [S]: Save calibration parameters & snapshot")
    print("  Key [Q]: Exit\n")

    try:
        frame_idx = 0
        while True:
            ret, frame, _ = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            fps = fps_meter.tick()

            # Always read trackbars live when display is active
            if has_display:
                active_range = read_trackbars(window_name)

            t0 = time.perf_counter()
            # 1. Convert BGR to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 2. Segment target pixels & clean noise
            mask = build_color_mask(hsv_frame, active_range)

            # 3. Extract centroid and bounding box
            centroid, bbox, area = extract_target_geometry(mask)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # 4. Annotate tracking overlay
            annotated = annotate_frame(frame, centroid, bbox, active_range.name, area)

            # 5. Composite split-view
            split_view = create_split_view(annotated, mask)

            # HUD telemetry overlay
            status_text = (
                f"FPS: {fps:5.1f} | Latency: {latency_ms:4.1f}ms | "
                f"Target: {active_range.name} | Area: {int(area)}px"
            )
            cv2.putText(
                split_view,
                status_text,
                (12, split_view.shape[0] - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 240, 255),
                1,
                cv2.LINE_AA,
            )

            hsv_text = (
                f"H: [{active_range.h_min:3d}, {active_range.h_max:3d}] | "
                f"S: [{active_range.s_min:3d}, {active_range.s_max:3d}] | "
                f"V: [{active_range.v_min:3d}, {active_range.v_max:3d}]"
                f"{' (Red Wrap)' if active_range.wrap_red else ''}"
            )
            cv2.putText(
                split_view,
                hsv_text,
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

                if key in (ord("q"), 27):
                    break
                elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                    preset_idx = key - ord("0")
                    active_range = COLOR_PRESETS[preset_idx]
                    update_trackbars(window_name, active_range)
                    print(f"[Preset] Activated: {active_range.name}")
                elif key == ord("s"):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = output_dir / f"color_{active_range.name.lower()}_{ts}.png"
                    json_path = output_dir / f"color_{active_range.name.lower()}_{ts}.json"

                    cv2.imwrite(str(img_path), split_view)
                    params = {
                        "color": active_range.name,
                        "h_min": active_range.h_min,
                        "h_max": active_range.h_max,
                        "s_min": active_range.s_min,
                        "s_max": active_range.s_max,
                        "v_min": active_range.v_min,
                        "v_max": active_range.v_max,
                        "wrap_red": active_range.wrap_red,
                    }
                    with open(json_path, "w") as f:
                        json.dump(params, f, indent=2)
                    print(f"[Saved] Snapshot: {img_path}")
                    print(f"[Saved] Config:   {json_path}")
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
                        f"Target: {active_range.name} | Area: {int(area)}px"
                    )
                    print(f"[Headless] {status}")
                if frame_idx >= 30:
                    out_path = output_dir / "headless_color_sample.png"
                    cv2.imwrite(str(out_path), split_view)
                    print(f"[Headless] Sample saved to: {out_path}. Exiting.")
                    break

    finally:
        cam.release()
        if has_display:
            cv2.destroyAllWindows()
        print("\n[Shutdown] Color detection pipeline released cleanly.")


if __name__ == "__main__":
    main()
