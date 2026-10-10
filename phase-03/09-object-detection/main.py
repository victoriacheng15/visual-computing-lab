"""Module 09: Object Detection, Multi-Scale Proposals & Non-Maximum Suppression (NMS).

Demonstrates the core mechanics of visual object detection without external model downloads:
multi-scale image pyramids, sliding-window response mapping, dense candidate proposal generation,
and vectorized Non-Maximum Suppression (NMS) via Intersection-over-Union (IoU).
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

from utils.camera import FPSMeter, ThreadedCamera  # noqa: E402


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute the Intersection over Union (IoU) of two bounding boxes.

    Boxes format: [x1, y1, x2, y2].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_w = max(0, x2 - x1)
    intersection_h = max(0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = float(area_a + area_b - intersection_area)

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.35,
    max_output: int = 50,
) -> list[int]:
    """Perform greedy Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        boxes: NumPy array of shape (N, 4) with coordinates [x1, y1, x2, y2].
        scores: NumPy array of shape (N,) with detection confidences.
        iou_threshold: Overlap ratio above which lower-scoring boxes are suppressed.
        max_output: Maximum number of retained detections.

    Returns:
        List of integer indices corresponding to retained boxes.
    """
    if len(boxes) == 0:
        return []

    # Sort indices in descending order by detection score
    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0 and len(keep) < max_output:
        current_idx = order[0]
        keep.append(int(current_idx))

        if len(order) == 1:
            break

        # Compute pairwise IoU between top box and all remaining candidate boxes
        remaining_indices = order[1:]
        current_box = boxes[current_idx]

        ious = np.array(
            [compute_iou(current_box, boxes[idx]) for idx in remaining_indices],
            dtype=np.float32,
        )

        # Retain only candidate boxes whose overlap is below the IoU threshold
        surviving = np.where(ious < iou_threshold)[0]
        order = remaining_indices[surviving]

    return keep


def generate_multiscale_proposals(
    frame_gray: np.ndarray,
    template_gray: np.ndarray,
    scales: list[float],
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate dense candidate bounding boxes across an image pyramid.

    Args:
        frame_gray: Single-channel grayscale input frame.
        template_gray: Single-channel grayscale target template.
        scales: List of scale factors to resize the template.
        confidence_threshold: Minimum correlation score to propose a box.

    Returns:
        Tuple of (boxes [N, 4], scores [N]).
    """
    frame_h, frame_w = frame_gray.shape[:2]
    candidate_boxes = []
    candidate_scores = []

    for scale in scales:
        scaled_w = int(template_gray.shape[1] * scale)
        scaled_h = int(template_gray.shape[0] * scale)

        if scaled_w < 20 or scaled_h < 20:
            continue
        if scaled_w >= frame_w or scaled_h >= frame_h:
            continue

        scaled_template = cv2.resize(
            template_gray, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
        )

        # Fast normalized cross-correlation response surface
        result = cv2.matchTemplate(frame_gray, scaled_template, cv2.TM_CCOEFF_NORMED)

        # Extract local peaks exceeding the confidence threshold
        y_locs, x_locs = np.where(result >= confidence_threshold)

        for pt_x, pt_y in zip(x_locs, y_locs, strict=False):
            score = float(result[pt_y, pt_x])
            candidate_boxes.append([pt_x, pt_y, pt_x + scaled_w, pt_y + scaled_h])
            candidate_scores.append(score)

    if not candidate_boxes:
        return np.empty((0, 4), dtype=np.int32), np.empty((0,), dtype=np.float32)

    return np.array(candidate_boxes, dtype=np.int32), np.array(candidate_scores, dtype=np.float32)


def create_hud_header(
    width: int,
    fps: float,
    mode_name: str,
    proposal_count: int,
    detection_count: int,
    conf_thresh: float,
    iou_thresh: float,
    has_target: bool,
) -> np.ndarray:
    """Create a dedicated top telemetry banner positioned above the video canvas."""
    hud_h = 90
    header = np.full((hud_h, width, 3), 20, dtype=np.uint8)

    # Status row
    target_status = "LOCKED" if has_target else "ACQUIRING (Center Crop)"
    target_color = (0, 255, 0) if has_target else (0, 165, 255)

    cv2.putText(
        header,
        f"MODE: {mode_name} | FPS: {fps:.1f} | TARGET: {target_status}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        target_color,
        1,
        cv2.LINE_AA,
    )

    # Telemetry metrics
    cv2.putText(
        header,
        f"Raw Proposals: {proposal_count}  |  NMS Detections: {detection_count}  |  "
        f"Confidence: {conf_thresh:.2f} (+/-)  |  IoU Thresh: {iou_thresh:.2f} ([/])",
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
        "[1] NMS  [2] Raw Storm  [3] Split  [Space] Retarget  [c] Cam  [s] Snap  [q] Quit",
        (15, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    # Border separating HUD from video
    cv2.line(header, (0, hud_h - 1), (width, hud_h - 1), (60, 60, 60), 1)

    return header


def main() -> None:
    """Main object detection and NMS visualization loop."""
    parser = argparse.ArgumentParser(description="Module 09: Object Detection & NMS Pipeline")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--conf", type=float, default=0.68, help="Initial confidence threshold")
    parser.add_argument("--iou", type=float, default=0.30, help="Initial IoU NMS threshold")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cam = ThreadedCamera(device_index=args.device, width=args.width, height=args.height).start()
    fps_meter = FPSMeter()

    conf_thresh = args.conf
    iou_thresh = args.iou
    scales = [0.7, 0.85, 1.0, 1.15, 1.3]

    view_mode = 1  # 1: NMS Clean, 2: Raw Proposals, 3: Side-by-side
    prev_view_mode = None
    target_template = None
    target_gray = None

    window_name = "Module 09: Object Detection & NMS Engine"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("=== Module 09: Object Detection Pipeline Initialized ===")
    print("Hold an object steady in the center and press [Space] to lock target template.")

    try:
        while True:
            ret, frame, _ = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            h, w = frame.shape[:2]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Auto-acquire default center target if uninitialized and camera warmed up
            if target_template is None and np.mean(frame) > 20:
                crop_size = min(w, h) // 4
                cx, cy = w // 2, h // 2
                target_template = frame[
                    cy - crop_size : cy + crop_size, cx - crop_size : cx + crop_size
                ].copy()
                target_gray = cv2.cvtColor(target_template, cv2.COLOR_BGR2GRAY)

            # Generate multi-scale proposals
            boxes, scores = np.empty((0, 4), dtype=np.int32), np.empty((0,), dtype=np.float32)
            kept_indices = []

            if target_gray is not None:
                boxes, scores = generate_multiscale_proposals(
                    frame_gray, target_gray, scales, conf_thresh
                )
                if len(boxes) > 0:
                    kept_indices = non_max_suppression(boxes, scores, iou_thresh)

            # Render detection canvases
            proposal_canvas = frame.copy()
            nms_canvas = frame.copy()

            # Render raw proposal storm (yellow translucent boxes)
            for box, score in zip(boxes, scores, strict=False):
                bx1, by1, bx2, by2 = box
                cv2.rectangle(proposal_canvas, (bx1, by1), (bx2, by2), (0, 255, 255), 1)

            # Render NMS filtered detections (thick green bounding boxes + confidence score)
            for idx in kept_indices:
                bx1, by1, bx2, by2 = boxes[idx]
                score = scores[idx]

                cv2.rectangle(nms_canvas, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                label = f"Target: {score * 100:.1f}%"
                cv2.rectangle(
                    nms_canvas,
                    (bx1, max(0, by1 - 20)),
                    (bx1 + 130, max(0, by1)),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    nms_canvas,
                    label,
                    (bx1 + 4, max(0, by1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Target preview in bottom right corner
            if target_template is not None:
                preview = cv2.resize(target_template, (80, 80))
                p_h, p_w = preview.shape[:2]
                pad = 10
                for canvas in (proposal_canvas, nms_canvas):
                    canvas[h - p_h - pad : h - pad, w - p_w - pad : w - pad] = preview
                    cv2.rectangle(
                        canvas,
                        (w - p_w - pad, h - p_h - pad),
                        (w - pad, h - pad),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        canvas,
                        "TARGET",
                        (w - p_w - pad + 5, h - p_h - pad - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                    )

            fps = fps_meter.tick()

            # Determine composite display based on view mode
            if view_mode == 1:
                display_frame = nms_canvas
                mode_str = "NMS Filtered Detections"
            elif view_mode == 2:
                display_frame = proposal_canvas
                mode_str = "Raw Candidate Proposal Storm"
            else:
                display_frame = np.hstack([proposal_canvas, nms_canvas])
                cv2.line(display_frame, (w, 0), (w, h), (255, 255, 255), 2)
                cv2.putText(
                    display_frame,
                    "RAW PROPOSALS",
                    (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    "NMS FILTERED",
                    (w + 15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                mode_str = "Split View (Raw vs NMS)"

            hud = create_hud_header(
                display_frame.shape[1],
                fps,
                mode_str,
                len(boxes),
                len(kept_indices),
                conf_thresh,
                iou_thresh,
                target_template is not None,
            )
            composite = np.vstack([hud, display_frame])

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
            elif key == ord(" "):
                # Re-capture target template from center box
                crop_size = min(w, h) // 4
                cx, cy = w // 2, h // 2
                target_template = frame[
                    cy - crop_size : cy + crop_size, cx - crop_size : cx + crop_size
                ].copy()
                target_gray = cv2.cvtColor(target_template, cv2.COLOR_BGR2GRAY)
                print("Target template captured from center frame.")
            elif key in (ord("+"), ord("=")):
                conf_thresh = min(0.95, conf_thresh + 0.02)
            elif key in (ord("-"), ord("_")):
                conf_thresh = max(0.30, conf_thresh - 0.02)
            elif key == ord("]"):
                iou_thresh = min(0.80, iou_thresh + 0.05)
            elif key == ord("["):
                iou_thresh = max(0.10, iou_thresh - 0.05)
            elif key == ord("c"):
                next_device = 2 if cam.device_index == 0 else 0
                print(f"Switching camera to device index {next_device}...")
                cam.release()
                cam = ThreadedCamera(
                    device_index=next_device, width=args.width, height=args.height
                ).start()
            elif key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = output_dir / f"detection_{timestamp}.png"
                cv2.imwrite(str(save_path), composite)
                print(f"Snapshot saved to: {save_path.name}")

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
