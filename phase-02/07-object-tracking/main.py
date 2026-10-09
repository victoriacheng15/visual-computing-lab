"""Module 07: Real-Time Object Tracking & State Estimation Lab.

Provides an interactive workbench for 2D Kalman filter state estimation,
sparse Lucas-Kanade corner tracking, dense Farneback optical flow, and CamShift.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure Module 02 camera engine and local tracker are importable
MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_02_DIR = REPO_ROOT / "phase-01" / "02-webcam"

for path in (MODULE_DIR, MODULE_02_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# OpenCV bundled Qt on Linux only provides libqxcb (XWayland)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from camera import FPSMeter, ThreadedCamera  # noqa: E402
from tracker import (  # noqa: E402
    CamShiftTracker,
    DenseOpticalFlowTracker,
    LinearKalman2D,
    SparseOpticalFlowTracker,
    TrackingMode,
    create_split_view,
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


def find_brightest_object(frame: np.ndarray) -> tuple[int, int] | None:
    """Isolate candidate target centroid under standard indoor lighting."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # Use 130 threshold so standard indoor items, phones, or hands trigger tracking
    _, thresh = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 07: Object Tracking & State Estimation")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    print("========================================")
    print(" Real-Time Object Tracking & State Lab")
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
    current_mode = TrackingMode.KALMAN

    # Initialize tracking engines
    kalman_tracker = LinearKalman2D(dt=1.0 / 30.0)
    sparse_lk = SparseOpticalFlowTracker(max_corners=80)
    dense_flow = DenseOpticalFlowTracker()
    camshift = CamShiftTracker()

    window_name = "Module 07: Object Tracking & State Estimation Lab"
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if has_display:
        cv2.namedWindow(window_name)

    print("\n[Controls]")
    print("  Key [1]: Kalman Filter (State estimation with occlusion recovery)")
    print("  Key [2]: Sparse Optical Flow (Lucas-Kanade point tracking)")
    print("  Key [3]: Dense Optical Flow (Farneback velocity vector field)")
    print("  Key [4]: CamShift (Adaptive color distribution tracking)")
    print("  Space  : Reset active tracker state or re-initialize target")
    print("  Key [C]: Cycle camera device")
    print("  Key [S]: Save snapshot & dump trajectory telemetry")
    print("  Key [Q]: Exit\n")

    simulated_occlusion = False
    last_state = None

    try:
        frame_idx = 0
        while True:
            ret, frame, _ = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            fps = fps_meter.tick()
            t0 = time.perf_counter()

            vis_left = frame.copy()
            vis_right = np.zeros_like(frame)

            match current_mode:
                case TrackingMode.KALMAN:
                    # Detect bright candidate object
                    raw_meas = find_brightest_object(frame)

                    # Simulate occlusion toggle
                    effective_meas = None if simulated_occlusion else raw_meas

                    # Update Kalman state
                    state = kalman_tracker.update(effective_meas)
                    last_state = state

                    ex, ey = state.estimated_pos
                    vx, vy = state.velocity

                    # Render trajectory path
                    for i in range(1, len(state.trajectory)):
                        cv2.line(
                            vis_left,
                            state.trajectory[i - 1],
                            state.trajectory[i],
                            (0, 255, 255),
                            2,
                        )

                    # Draw measurement (red) vs estimate (green)
                    if effective_meas is not None:
                        mx, my = effective_meas
                        cv2.circle(vis_left, (mx, my), 7, (0, 0, 255), 2)  # Measurement
                    cv2.circle(vis_left, (ex, ey), 10, (0, 255, 0), -1)  # Kalman Estimate

                    # Velocity vector arrow
                    arrow_x = int(ex + vx * 5.0)
                    arrow_y = int(ey + vy * 5.0)
                    cv2.arrowedLine(
                        vis_left, (ex, ey), (arrow_x, arrow_y), (255, 0, 0), 2, tipLength=0.3
                    )

                    # Right telemetry canvas: State phase space (Position + Velocity)
                    vis_right = np.full_like(frame, 24)  # Dark slate background

                    # Draw subtle coordinate grid lines
                    h_r, w_r = vis_right.shape[:2]
                    for gx in range(0, w_r, 80):
                        cv2.line(vis_right, (gx, 0), (gx, h_r), (38, 38, 38), 1)
                    for gy in range(0, h_r, 80):
                        cv2.line(vis_right, (0, gy), (w_r, gy), (38, 38, 38), 1)

                    # Radar mini-map center
                    radar_cx = w_r - 180
                    radar_cy = 160
                    cv2.circle(vis_right, (radar_cx, radar_cy), 80, (50, 50, 50), 1)
                    # Draw crosshairs
                    pt_left = (radar_cx - 85, radar_cy)
                    pt_right = (radar_cx + 85, radar_cy)
                    pt_top = (radar_cx, radar_cy - 85)
                    pt_bottom = (radar_cx, radar_cy + 85)
                    cv2.line(vis_right, pt_left, pt_right, (60, 60, 60), 1)
                    cv2.line(vis_right, pt_top, pt_bottom, (60, 60, 60), 1)
                    cv2.putText(
                        vis_right,
                        "Velocity Radar",
                        (radar_cx - 50, radar_cy + 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (140, 140, 140),
                        1,
                    )

                    # Draw velocity needle on radar
                    needle_x = int(radar_cx + np.clip(vx * 2.5, -75, 75))
                    needle_y = int(radar_cy + np.clip(vy * 2.5, -75, 75))
                    cv2.arrowedLine(
                        vis_right,
                        (radar_cx, radar_cy),
                        (needle_x, needle_y),
                        (0, 255, 255),
                        2,
                        tipLength=0.25,
                    )
                    cv2.circle(vis_right, (needle_x, needle_y), 4, (0, 255, 255), -1)

                    status_str = (
                        "OCCLUDED (Predicting)" if state.is_occluded else "TRACKING (Active)"
                    )
                    color_status = (0, 80, 255) if state.is_occluded else (0, 255, 0)

                    cv2.putText(
                        vis_right,
                        f"State: {status_str}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color_status,
                        2,
                    )
                    cv2.putText(
                        vis_right,
                        f"Estimated Pos : ({ex:4d}, {ey:4d})",
                        (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        vis_right,
                        f"Velocity Vector: ({vx:+5.1f}, {vy:+5.1f}) px/f",
                        (30, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        1,
                    )
                    speed = np.hypot(vx, vy)
                    cv2.putText(
                        vis_right,
                        f"Kinematic Speed: {speed:5.1f} px/f",
                        (30, 175),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 255, 200),
                        1,
                    )
                    cv2.putText(
                        vis_right,
                        f"Trajectory Pts: {len(state.trajectory)}",
                        (30, 215),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 200, 200),
                        1,
                    )
                    cv2.putText(
                        vis_right,
                        "Press 'o' to toggle occlusion simulation",
                        (30, 270),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (150, 255, 150),
                        1,
                    )

                case TrackingMode.LUCAS_KANADE:
                    vis_left, vis_right = sparse_lk.process(frame)

                case TrackingMode.FARNEBACK:
                    vis_right = dense_flow.process(frame)

                case TrackingMode.CAMSHIFT:
                    if camshift.track_window is None:
                        # Auto-seed initial window at center
                        h, w = frame.shape[:2]
                        seed_bbox = (w // 2 - 50, h // 2 - 50, 100, 100)
                        camshift.initialize(frame, seed_bbox)

                    vis_left, backproj = camshift.process(frame)
                    if backproj is not None:
                        vis_right = cv2.cvtColor(backproj, cv2.COLOR_GRAY2BGR)

            latency_ms = (time.perf_counter() - t0) * 1000.0

            split_view = create_split_view(
                vis_left,
                vis_right,
                left_title=f"Mode: {current_mode.value}",
                right_title="State / Flow Telemetry",
            )

            # HUD Line 1: Performance
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

            # HUD Line 2: Control hints
            cv2.putText(
                split_view,
                "Keys 1-4: Modes | Space: Reset | 'o': Toggle Occlusion | 'c': Camera | 's': Save",
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

                mode_map = {
                    ord("1"): (TrackingMode.KALMAN, "Kalman Filter"),
                    ord("2"): (TrackingMode.LUCAS_KANADE, "Lucas-Kanade"),
                    ord("3"): (TrackingMode.FARNEBACK, "Dense Farneback"),
                    ord("4"): (TrackingMode.CAMSHIFT, "CamShift"),
                }

                match key:
                    case 27 | 113:  # ESC or 'q'
                        break

                    case k if k in mode_map:
                        current_mode, mode_name = mode_map[k]
                        print(f"[Mode] Activated: {mode_name}")

                    case 32:  # Space (reset)
                        kalman_tracker.reset()
                        sparse_lk.reset()
                        dense_flow.reset()
                        camshift.reset()
                        print("[Tracking] Active tracker state reset.")

                    case 111:  # 'o' toggle occlusion simulation
                        simulated_occlusion = not simulated_occlusion
                        status = "OCCLUDED" if simulated_occlusion else "VISIBLE"
                        print(f"[Kalman] Occlusion simulation toggled: {status}")

                    case 115:  # 's' snapshot & telemetry export
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        slug = current_mode.name.lower()
                        img_path = output_dir / f"track_{slug}_{ts}.png"
                        json_path = output_dir / f"track_{slug}_{ts}.json"

                        cv2.imwrite(str(img_path), split_view)
                        telemetry = {
                            "mode": current_mode.value,
                            "fps": round(fps, 1),
                            "latency_ms": round(latency_ms, 2),
                            "timestamp": ts,
                        }
                        if last_state is not None:
                            telemetry["kalman"] = {
                                "estimated_pos": last_state.estimated_pos,
                                "velocity": [round(v, 2) for v in last_state.velocity],
                                "is_occluded": last_state.is_occluded,
                                "trajectory_length": len(last_state.trajectory),
                            }
                        with open(json_path, "w") as f:
                            json.dump(telemetry, f, indent=2)
                        print(f"[Saved] Snapshot:  {img_path}")
                        print(f"[Saved] Telemetry: {json_path}")

                    case 99:  # 'c' cycle camera
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
