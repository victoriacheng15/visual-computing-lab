"""Module 08: Real-Time Feature Detection, Matching & Planar Homography Lab.

Provides an interactive workbench for ORB vs SIFT feature matching,
Lowe's ratio test, RANSAC planar homography, and augmented reality perspective warping.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure Module 02 camera engine and local matcher are importable
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
from matcher import (  # noqa: E402
    FeatureMode,
    FeaturePipeline,
    create_split_view,
    render_feature_visualization,
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


def create_synthetic_ar_asset(width: int = 400, height: int = 300) -> np.ndarray:
    """Generate a clean synthetic cyberpunk AR graphic billboard."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Gradient backdrop
    for y in range(height):
        ratio = y / height
        img[y, :] = (int(30 + 100 * ratio), int(10 + 40 * ratio), int(60 + 120 * ratio))

    # Grid lines
    for x in range(0, width, 40):
        cv2.line(img, (x, 0), (x, height), (0, 200, 255), 1)
    for y in range(0, height, 40):
        cv2.line(img, (0, y), (width, y), (0, 200, 255), 1)

    # Glowing emblem
    cx, cy = width // 2, height // 2
    cv2.circle(img, (cx, cy), 50, (0, 255, 200), -1)
    cv2.circle(img, (cx, cy), 65, (0, 200, 255), 3)

    cv2.putText(
        img, "AR SYNTHESIS", (cx - 85, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2
    )
    cv2.putText(
        img,
        "HOMOGRAPHY LOCKED",
        (cx - 95, cy + 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 08: Feature Detection & Planar Homography")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    print("========================================")
    print(" Real-Time Planar Homography & AR Lab")
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
    current_mode = FeatureMode.ORB

    pipeline = FeaturePipeline(max_features=800)
    ar_asset = create_synthetic_ar_asset(400, 300)

    window_name = "Module 08: Feature Detection & Homography Lab"
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if has_display:
        cv2.namedWindow(window_name)

    print("\n[Controls]")
    print("  Key [1]: ORB Matching & Homography (Fast binary descriptors)")
    print("  Key [2]: SIFT Matching & Homography (High-precision gradient descriptors)")
    print("  Key [3]: Keypoints Only (Inspect raw corners, scales, and angles)")
    print("  Key [4]: Augmented Reality Warp (Virtual billboard on target)")
    print("  Space  : Capture center crop as new reference planar target")
    print("  Key [C]: Cycle camera device")
    print("  Key [S]: Save snapshot & dump homography telemetry")
    print("  Key [Q]: Exit\n")

    last_result = None

    try:
        frame_idx = 0
        while True:
            ret, frame, _ = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            fps = fps_meter.tick()
            t0 = time.perf_counter()

            # Initialize reference target once camera has warmed up and adjusted exposure
            if pipeline.ref_image is None and np.mean(frame) > 20:
                h_f, w_f = frame.shape[:2]
                box_sz = 260
                cx, cy = w_f // 2, h_f // 2
                half = box_sz // 2
                crop = frame[cy - half : cy + half, cx - half : cx + half]
                pipeline.set_reference_image(crop)
                print("[Target] Auto-acquired exposed center region as reference target.")

            scene_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            use_sift = current_mode == FeatureMode.SIFT

            h_result, scene_kps, _ = pipeline.find_homography(
                scene_gray,
                use_sift=use_sift,
                ratio_threshold=0.75,
                ransac_reproj_thresh=3.0,
            )
            last_result = h_result

            vis_left, vis_right = render_feature_visualization(
                frame,
                pipeline,
                current_mode,
                h_result,
                scene_kps,
                ar_overlay_image=ar_asset,
            )

            latency_ms = (time.perf_counter() - t0) * 1000.0

            split_view = create_split_view(
                vis_left,
                vis_right,
                left_title=f"Mode: {current_mode.value}",
                right_title="Feature Telemetry & Target",
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

            # HUD Line 2: Hints
            hint_text = (
                "Keys 1-4: Modes | Space: Capture Target | 'c': Camera | 's': Save | 'q': Exit"
            )
            cv2.putText(
                split_view,
                hint_text,
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
                    ord("1"): (FeatureMode.ORB, "ORB Descriptors"),
                    ord("2"): (FeatureMode.SIFT, "SIFT Descriptors"),
                    ord("3"): (FeatureMode.KEYPOINTS_ONLY, "Keypoints Only"),
                    ord("4"): (FeatureMode.AR_WARP, "Augmented Reality Warp"),
                }

                match key:
                    case 27 | 113:  # ESC or 'q'
                        break

                    case k if k in mode_map:
                        current_mode, mode_name = mode_map[k]
                        print(f"[Mode] Activated: {mode_name}")

                    case 32:  # Space - capture new reference target
                        h_f, w_f = frame.shape[:2]
                        box_sz = 260
                        cx, cy = w_f // 2, h_f // 2
                        new_crop = frame[
                            cy - box_sz // 2 : cy + box_sz // 2, cx - box_sz // 2 : cx + box_sz // 2
                        ]
                        pipeline.set_reference_image(new_crop)
                        print("[Target] Captured new center region as reference target.")

                    case 115:  # 's' snapshot & homography dump
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        slug = current_mode.name.lower()
                        img_path = output_dir / f"homography_{slug}_{ts}.png"
                        json_path = output_dir / f"homography_{slug}_{ts}.json"

                        cv2.imwrite(str(img_path), split_view)
                        telemetry = {
                            "mode": current_mode.value,
                            "fps": round(fps, 1),
                            "latency_ms": round(latency_ms, 2),
                            "timestamp": ts,
                        }
                        if last_result is not None and last_result.homography is not None:
                            telemetry["homography_matrix"] = last_result.homography.tolist()
                            telemetry["inlier_count"] = last_result.inlier_count
                            telemetry["total_matches"] = last_result.total_matches

                        with open(json_path, "w") as f:
                            json.dump(telemetry, f, indent=2)
                        print(f"[Saved] Snapshot:   {img_path}")
                        print(f"[Saved] Homography: {json_path}")

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
