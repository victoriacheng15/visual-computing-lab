"""Module 02: Real-time Camera Ingestion & Latency Profiling.

Runs the thread-isolated capture engine and renders an interactive telemetry HUD
monitoring FPS, resolution, and ingestion latency.
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

# OpenCV bundled Qt on Linux only provides libqxcb (XWayland)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
from camera import FPSMeter, ThreadedCamera


def detect_camera_indices() -> list[int]:
    """Scan and return device indices that actually stream video (excluding metadata nodes)."""
    valid_devices = []
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
                valid_devices.append(idx)

    return valid_devices if valid_devices else [0]


def draw_hud(
    frame: np.ndarray,
    fps: float,
    lag_ms: float,
    backend: str,
    resolution: tuple[int, int],
    device_index: int,
) -> np.ndarray:
    """Render a non-intrusive heads-up display overlay on the video frame."""
    hud = frame.copy()
    w, h = resolution

    # Semi-transparent dark banner at top
    overlay = hud.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.75, hud, 0.25, 0, hud)

    # Telemetry text
    status_text = (
        f"FPS: {fps:5.1f} | Ingest Lag: {lag_ms:4.1f}ms | "
        f"Dev: {device_index} | Res: {w}x{h} | {backend}"
    )
    cv2.putText(
        hud,
        status_text,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 240, 255),
        1,
        cv2.LINE_AA,
    )

    # Footer instructions
    cv2.putText(
        hud,
        "[Q] Quit   [S] Snapshot   [C] Switch Camera   [T] Toggle Synthetic",
        (12, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return hud


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 02: Real-time Camera Ingestion")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Camera device index (e.g. 0 for Integrated Camera, 2 for MX Brio)",
    )
    args = parser.parse_args()

    print("========================================")
    print(" Visual Computing Lab - Module 02")
    print(" Real-Time Camera Ingestion Pipeline")
    print("========================================")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    available_cams = detect_camera_indices()
    current_device = args.device
    use_synthetic = False

    print(f"\n[Hardware] Detected video devices: {available_cams}")
    print(f"[Hardware] Starting with camera device: {current_device}")

    cam = ThreadedCamera(
        device_index=current_device,
        width=1280,
        height=720,
        fps=30,
        use_synthetic=use_synthetic,
    )
    cam.start()

    fps_meter = FPSMeter(window_size=30)
    window_name = "Module 02: Threaded Camera Feed & Telemetry HUD"

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    print(f"\n[Ingestion] Running backend: {cam.backend_name}")
    print("[Ingestion] Keys: [C] switch cam, [S] snapshot, [T] synthetic, [Q] quit.\n")

    try:
        frame_idx = 0
        while True:
            ret, frame, capture_time = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            current_time = time.perf_counter()
            fps = fps_meter.tick()
            lag_ms = (current_time - capture_time) * 1000.0
            resolution = cam.get_resolution()

            annotated_frame = draw_hud(
                frame, fps, lag_ms, cam.backend_name, resolution, cam.device_index
            )

            if has_display:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):  # 'q' or Esc
                    break
                elif key == ord("s"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = output_dir / f"snapshot_dev{cam.device_index}_{timestamp}.png"
                    cv2.imwrite(str(out_path), frame)
                    print(f"[Snapshot] Saved to: {out_path}")
                elif key == ord("c"):
                    # Cycle to the next available hardware camera
                    if len(available_cams) > 1:
                        next_idx = (available_cams.index(current_device) + 1) % len(available_cams)
                        current_device = available_cams[next_idx]
                    else:
                        current_device = 2 if current_device == 0 else 0
                    print(f"\n[Switch] Reconnecting to camera device {current_device}...")
                    cam.release()
                    cam = ThreadedCamera(
                        device_index=current_device,
                        width=1280,
                        height=720,
                        fps=30,
                        use_synthetic=False,
                    )
                    cam.start()
                    use_synthetic = cam.use_synthetic
                elif key == ord("t"):
                    # Toggle synthetic mode
                    use_synthetic = not use_synthetic
                    print(f"\n[Mode] Switching use_synthetic = {use_synthetic}")
                    cam.release()
                    cam = ThreadedCamera(
                        device_index=current_device,
                        width=1280,
                        height=720,
                        fps=30,
                        use_synthetic=use_synthetic,
                    )
                    cam.start()
            else:
                # Headless verification: process 30 frames and save sample
                frame_idx += 1
                if frame_idx % 10 == 0:
                    status = f"Frame {frame_idx:03d} | FPS: {fps:.1f} | Lag: {lag_ms:.2f}ms"
                    print(f"[Headless] {status}")
                if frame_idx >= 30:
                    out_path = output_dir / "headless_sample.png"
                    cv2.imwrite(str(out_path), annotated_frame)
                    print(f"[Headless] Sample saved to: {out_path}. Exiting verification.")
                    break

    finally:
        cam.release()
        if has_display:
            cv2.destroyAllWindows()
        print("\n[Shutdown] Camera pipeline released cleanly.")


if __name__ == "__main__":
    main()
