"""Module 11: Real-Time Foreground Segmentation & Virtual Background Replacement.

Demonstrates the core mechanics of visual segmentation without external model downloads:
Mixture of Gaussians (MOG2) vs K-Nearest Neighbors (KNN) background subtraction,
shadow suppression, morphological mask cleanup, alpha feathering, and real-time
virtual background replacement (chroma green-screen and portrait blur).
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
import onnxruntime as ort  # noqa: E402

from utils.camera import FPSMeter, ThreadedCamera  # noqa: E402


def run_neural_segmentation(
    session: ort.InferenceSession,
    frame: np.ndarray,
    input_name: str,
) -> np.ndarray:
    """Infer deep portrait segmentation mask using U2-NetP ONNX model.

    Returns:
        Float32 alpha mask of shape (H, W) in [0.0, 1.0].
    """
    h, w = frame.shape[:2]
    # Resize to model input 320x320 and normalize
    resized = cv2.resize(frame, (320, 320)).astype(np.float32) / 255.0
    # ImageNet standard normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    resized = (resized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = np.transpose(resized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    preds = session.run(None, {input_name: tensor})
    pred = preds[0][0, 0]
    # Min-max normalization of output saliency map
    p_min, p_max = float(pred.min()), float(pred.max())
    pred_norm = (pred - p_min) / (p_max - p_min + 1e-8)
    alpha = cv2.resize(pred_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    # Threshold slight background noise
    alpha = np.clip((alpha - 0.2) / 0.8, 0.0, 1.0)
    return alpha.astype(np.float32)


def generate_virtual_studio_backdrop(height: int, width: int) -> np.ndarray:
    """Generate a clean procedural studio background with ambient illumination."""
    backdrop = np.zeros((height, width, 3), dtype=np.uint8)

    # Vertical gradient (deep navy to warm ambient studio tone)
    for y in range(height):
        ratio = y / max(1, height)
        b = int(120 * (1.0 - ratio) + 40 * ratio)
        g = int(70 * (1.0 - ratio) + 30 * ratio)
        r = int(50 * (1.0 - ratio) + 20 * ratio)
        backdrop[y, :] = [b, g, r]

    # Warm ambient light orb
    cv2.circle(backdrop, (width // 2, height // 3), int(width * 0.35), (150, 110, 70), -1)

    # Secondary cool accent light
    accent_pos = (int(width * 0.8), int(height * 0.7))
    cv2.circle(backdrop, accent_pos, int(width * 0.25), (100, 60, 40), -1)

    # Heavy blur for photographic depth-of-field effect
    backdrop = cv2.GaussianBlur(backdrop, (101, 101), 0)

    # Subtle shelf geometric accents
    shelf_y = int(height * 0.65)
    cv2.line(backdrop, (0, shelf_y), (width, shelf_y), (40, 30, 25), 3)

    return backdrop


def refine_foreground_mask(
    raw_mask: np.ndarray,
    kernel_size: int = 5,
    feather_sigma: float = 3.0,
) -> np.ndarray:
    """Clean sensor noise, exclude shadows, and generate soft alpha matte.

    Args:
        raw_mask: Single-channel mask from background subtractor.
        kernel_size: Size of morphological structuring element.
        feather_sigma: Standard deviation for Gaussian alpha edge feathering.

    Returns:
        Float32 alpha matte normalized to [0.0, 1.0].
    """
    # Exclude shadows (MOG2/KNN marks shadows with gray value ~127, foreground is 255)
    _, binary_mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)

    # Morphological Opening (Erosion -> Dilation) to eliminate isolated false-positive noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Morphological Closing (Dilation -> Erosion) to fill interior foreground holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Contour filtering: retain only significant connected components (>800 px)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(closed)
    for c in contours:
        if cv2.contourArea(c) >= 800:
            cv2.drawContours(filtered_mask, [c], -1, 255, -1)

    # Gaussian feathering to generate smooth, natural transition boundaries
    feather_ksize = max(3, int(feather_sigma * 4) | 1)
    feathered = cv2.GaussianBlur(filtered_mask, (feather_ksize, feather_ksize), feather_sigma)

    return feathered.astype(np.float32) / 255.0


def composite_frame(
    foreground: np.ndarray,
    background: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Alpha blend foreground and background tensors: I = alpha * FG + (1 - alpha) * BG."""
    alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    blended = alpha_3d * foreground.astype(np.float32) + (1.0 - alpha_3d) * background.astype(
        np.float32
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def create_hud_header(
    width: int,
    fps: float,
    mode_name: str,
    subtractor_name: str,
    learning_rate: float,
    feather_sigma: float,
    fg_ratio: float,
) -> np.ndarray:
    """Render top telemetry header positioned strictly above the video."""
    hud_h = 90
    header = np.full((hud_h, width, 3), 20, dtype=np.uint8)

    bg_status = "LOCKED (FROZEN)" if learning_rate == 0.0 else "ADAPTING (AUTO)"
    status_color = (0, 255, 0) if learning_rate == 0.0 else (0, 165, 255)

    # Status row
    cv2.putText(
        header,
        f"MODE: {mode_name} | BG: {bg_status} | FPS: {fps:.1f}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        1,
        cv2.LINE_AA,
    )

    # Telemetry metrics
    if learning_rate == 0.0:
        lr_str = "FROZEN"
    elif learning_rate < 0:
        lr_str = "AUTO"
    else:
        lr_str = f"{learning_rate:.3f}"
    cv2.putText(
        header,
        f"Foreground: {fg_ratio * 100:.1f}%  |  Model: {subtractor_name}  |  "
        f"Rate: {lr_str}  |  Feather: {feather_sigma:.1f}px ([/])",
        (15, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Key controls
    cv2.putText(
        header,
        "[Space] Lock Clean BG  [r] Relearn  [1-5] View  [m] Model  [s] Snap  [q] Quit",
        (15, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    # Border separating HUD from video
    cv2.line(header, (0, hud_h - 1), (width, hud_h - 1), (60, 60, 60), 1)

    return header


def main() -> None:
    """Main foreground segmentation and virtual background replacement loop."""
    parser = argparse.ArgumentParser(description="Module 11: Real-Time Virtual Background")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cam = ThreadedCamera(device_index=args.device, width=args.width, height=args.height).start()
    fps_meter = FPSMeter()

    # Initialize ONNX Neural portrait segmentation model
    model_path = Path(__file__).resolve().parent / "u2netp.onnx"
    neural_session = None
    neural_input_name = None
    if model_path.exists():
        neural_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        neural_input_name = neural_session.get_inputs()[0].name

    # Initialize classical background subtractors for comparison
    subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )
    subtractor_mog2.setShadowThreshold(0.5)
    subtractor_knn = cv2.createBackgroundSubtractorKNN(
        history=500, dist2Threshold=600.0, detectShadows=True
    )
    subtractor_knn.setShadowThreshold(0.5)

    subtractor_name = "Neural (U2-NetP)" if neural_session else "MOG2 (Gaussian Mixture)"
    active_subtractor = subtractor_mog2

    view_mode = 1  # 1: Studio, 2: Green Screen, 3: Portrait Blur, 4: Mask, 5: Split
    prev_view_mode = None

    learning_rate = 0.002
    feather_sigma = 3.0

    studio_backdrop = generate_virtual_studio_backdrop(args.height, args.width)
    green_screen = np.full((args.height, args.width, 3), (0, 255, 0), dtype=np.uint8)

    window_name = "Module 11: Segmentation & Virtual Background"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("=== Module 11: Virtual Background Pipeline Initialized ===")
    print("Step away for 1-2 seconds to allow background calibration, then step into frame.")

    try:
        while True:
            ret, frame, _ = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            h, w = frame.shape[:2]
            fps = fps_meter.tick()

            # Ensure backdrops match frame dimensions if camera negotiated different size
            if studio_backdrop.shape[:2] != (h, w):
                studio_backdrop = generate_virtual_studio_backdrop(h, w)
                green_screen = np.full((h, w, 3), (0, 255, 0), dtype=np.uint8)

            # Compute foreground alpha matte
            if "Neural" in subtractor_name and neural_session is not None:
                alpha = run_neural_segmentation(neural_session, frame, neural_input_name)
            else:
                # Apply classical background subtractor (MOG2 / KNN)
                blurred_input = cv2.GaussianBlur(frame, (5, 5), 0)
                raw_mask = active_subtractor.apply(blurred_input, learningRate=learning_rate)
                alpha = refine_foreground_mask(raw_mask, kernel_size=5, feather_sigma=feather_sigma)
            fg_ratio = float(np.mean(alpha))

            # Compose active visual output based on view mode
            if view_mode == 1:
                display_frame = composite_frame(frame, studio_backdrop, alpha)
                mode_str = "Virtual Studio Backdrop"
            elif view_mode == 2:
                display_frame = composite_frame(frame, green_screen, alpha)
                mode_str = "Chroma Green Screen"
            elif view_mode == 3:
                # Heavy Gaussian blur applied to raw video background
                blurred_bg = cv2.GaussianBlur(frame, (51, 51), 0)
                display_frame = composite_frame(frame, blurred_bg, alpha)
                mode_str = "Portrait Background Blur"
            elif view_mode == 4:
                # Alpha mask visualization (white = foreground, black = background)
                mask_vis = (alpha * 255.0).astype(np.uint8)
                display_frame = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
                mode_str = "Refined Alpha Matte"
            else:
                # Split view: Raw camera feed on left, Studio replacement on right
                composite_studio = composite_frame(frame, studio_backdrop, alpha)
                display_frame = np.hstack([frame, composite_studio])
                cv2.line(display_frame, (w, 0), (w, h), (255, 255, 255), 2)
                cv2.putText(
                    display_frame,
                    "RAW CAMERA FEED",
                    (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    "VIRTUAL REPLACEMENT",
                    (w + 15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                mode_str = "Split View (Raw vs Virtual)"

            # Construct top telemetry HUD header positioned above the video
            hud = create_hud_header(
                display_frame.shape[1],
                fps,
                mode_str,
                subtractor_name,
                learning_rate,
                feather_sigma,
                fg_ratio,
            )
            composite = np.vstack([hud, display_frame])

            # Resize window on mode transitions
            if view_mode != prev_view_mode:
                cv2.resizeWindow(window_name, composite.shape[1], composite.shape[0])
                prev_view_mode = view_mode

            cv2.imshow(window_name, composite)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("1"):
                view_mode = 1
            elif key == ord("2"):
                view_mode = 2
            elif key == ord("3"):
                view_mode = 3
            elif key == ord("4"):
                view_mode = 4
            elif key == ord("5"):
                view_mode = 5
            elif key == ord("m"):
                # Cycle between Neural (U2-NetP), MOG2, and KNN models
                if "Neural" in subtractor_name:
                    active_subtractor = subtractor_mog2
                    subtractor_name = "MOG2 (Gaussian Mixture)"
                elif "MOG2" in subtractor_name:
                    active_subtractor = subtractor_knn
                    subtractor_name = "KNN (K-Nearest Neighbors)"
                else:
                    if neural_session is not None:
                        subtractor_name = "Neural (U2-NetP)"
                    else:
                        active_subtractor = subtractor_mog2
                        subtractor_name = "MOG2 (Gaussian Mixture)"
                print(f"Switched segmentation model to: {subtractor_name}")
            elif key == ord(" "):
                # Train 60 frames on current background and set rate = 0.002
                print("Calibrating clean background plate (60 frames)... stay out of frame.")
                subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=50, detectShadows=True
                )
                subtractor_mog2.setShadowThreshold(0.5)
                subtractor_knn = cv2.createBackgroundSubtractorKNN(
                    history=500, dist2Threshold=600.0, detectShadows=True
                )
                subtractor_knn.setShadowThreshold(0.5)
                active_subtractor = subtractor_knn if "KNN" in subtractor_name else subtractor_mog2
                for _ in range(60):
                    ret_c, frame_c, _ = cam.read()
                    if ret_c and frame_c is not None:
                        bl_c = cv2.GaussianBlur(frame_c, (5, 5), 0)
                        active_subtractor.apply(bl_c, learningRate=0.05)
                        time.sleep(0.015)
                learning_rate = 0.0
                print("Clean background LOCKED (rate=0.0). Step in: you will stay visible!")
            elif key == ord("r"):
                # Reset background model to adaptive learning
                subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=25, detectShadows=True
                )
                subtractor_knn = cv2.createBackgroundSubtractorKNN(
                    history=500, dist2Threshold=400.0, detectShadows=True
                )
                active_subtractor = subtractor_knn if "KNN" in subtractor_name else subtractor_mog2
                learning_rate = -1.0
                print("Background reset to continuous adaptive learning mode.")
            elif key in (ord("+"), ord("=")):
                learning_rate = 0.005 if learning_rate < 0 else min(0.1, learning_rate + 0.005)
                print(f"Learning rate set to: {learning_rate:.3f}")
            elif key in (ord("-"), ord("_")):
                if learning_rate <= 0.005:
                    learning_rate = -1.0
                else:
                    learning_rate = max(0.001, learning_rate - 0.005)
                lr_str = "AUTO" if learning_rate < 0 else f"{learning_rate:.3f}"
                print(f"Learning rate set to: {lr_str}")
            elif key == ord("]"):
                feather_sigma = min(12.0, feather_sigma + 0.5)
                print(f"Alpha feathering sigma set to: {feather_sigma:.1f}px")
            elif key == ord("["):
                feather_sigma = max(0.5, feather_sigma - 0.5)
                print(f"Alpha feathering sigma set to: {feather_sigma:.1f}px")
            elif key == ord("c"):
                next_device = 2 if cam.device_index == 0 else 0
                print(f"Switching camera to device index {next_device}...")
                cam.release()
                cam = ThreadedCamera(
                    device_index=next_device, width=args.width, height=args.height
                ).start()
            elif key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = output_dir / f"segmentation_{timestamp}.png"
                cv2.imwrite(str(save_path), composite)
                print(f"Snapshot saved to: {save_path.name}")

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
