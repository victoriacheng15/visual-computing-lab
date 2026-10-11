"""Module 10: Real-Time Visual Feature Embedding & Image Classification.

Demonstrates the core mechanics of vision classifiers without external model downloads:
dense feature embedding extraction, cosine similarity metrics, Softmax probability distributions,
top-k ranking, and temporal Exponential Moving Average (EMA) prediction smoothing.
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


def extract_feature_embedding(image: np.ndarray) -> np.ndarray:
    """Extract a normalized 128-dimensional spatial and photometric feature embedding.

    Combines multi-scale color distributions, HSV channel histograms,
    spatial gradient orientation energy, and quadrant texture statistics.
    """
    img_resized = cv2.resize(image, (160, 160))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    features = []

    # 1. Global channel statistics (mean and std across BGR and HSV) -> 12 dims
    for channel in cv2.split(img_resized):
        features.extend([float(np.mean(channel)), float(np.std(channel))])
    for channel in cv2.split(hsv):
        features.extend([float(np.mean(channel)), float(np.std(channel))])

    # 2. 32-bin Hue-Saturation color distribution histogram -> 32 dims
    hist_hs = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
    cv2.normalize(hist_hs, hist_hs)
    features.extend(hist_hs.flatten().tolist())

    # 3. Spatial gradient energy across 4 spatial quadrants -> 32 dims
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    angle = (np.arctan2(sobel_y, sobel_x) * (180.0 / np.pi)) % 180.0

    quadrants = [
        (slice(0, 80), slice(0, 80)),
        (slice(0, 80), slice(80, 160)),
        (slice(80, 160), slice(0, 80)),
        (slice(80, 160), slice(80, 160)),
    ]

    for q_y, q_x in quadrants:
        q_mag = magnitude[q_y, q_x]
        q_ang = angle[q_y, q_x]
        # 8-bin orientation histogram weighted by gradient magnitude
        hist_ang, _ = np.histogram(q_ang, bins=8, range=(0.0, 180.0), weights=q_mag)
        total_energy = float(np.sum(hist_ang)) + 1e-6
        features.extend((hist_ang / total_energy).tolist())

    # 4. Central crop focus embedding vs peripheral context -> 26 dims
    center_crop = gray[40:120, 40:120]
    features.append(float(np.mean(center_crop)))
    features.append(float(np.std(center_crop)))

    # Laplacian high-frequency edge variance
    lap = cv2.Laplacian(center_crop, cv2.CV_32F)
    features.append(float(np.var(lap)))

    # 8-bin grayscale intensity distribution in central focus
    hist_center, _ = np.histogram(center_crop, bins=8, range=(0, 256), density=True)
    features.extend(hist_center.tolist())

    # Remaining padding features to reach exactly 128 dimensions -> 15 dims
    features.extend([float(np.percentile(center_crop, p)) for p in range(10, 100, 20)])
    features.extend([float(np.percentile(gray, p)) for p in range(10, 100, 10)])

    embedding = np.array(features[:128], dtype=np.float32)

    # L2-normalization for cosine distance metric
    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding /= norm

    return embedding


def compute_softmax(logits: np.ndarray, temperature: float = 0.15) -> np.ndarray:
    """Compute temperature-scaled Softmax probability distribution."""
    scaled = logits / max(0.01, temperature)
    exp_scores = np.exp(scaled - np.max(scaled))
    return exp_scores / (np.sum(exp_scores) + 1e-9)


class EmbeddingClassifier:
    """Few-shot interactive nearest-centroid feature classifier."""

    def __init__(self, num_classes: int = 4) -> None:
        self.num_classes = num_classes
        self.class_names = [
            "Class 1 (Background)",
            "Class 2 (Object A)",
            "Class 3 (Object B)",
            "Class 4 (Object C)",
        ]
        self.centroids: list[np.ndarray | None] = [None] * num_classes
        self.sample_counts: list[int] = [0] * num_classes

    def add_sample(self, class_idx: int, embedding: np.ndarray) -> None:
        """Add sample embedding and update class centroid running average."""
        if not 0 <= class_idx < self.num_classes:
            return

        current_cnt = self.sample_counts[class_idx]
        current_centroid = self.centroids[class_idx]

        if current_centroid is None or current_cnt == 0:
            self.centroids[class_idx] = embedding.copy()
            self.sample_counts[class_idx] = 1
        else:
            # Incremental centroid update
            updated = (current_centroid * current_cnt + embedding) / (current_cnt + 1)
            updated /= np.linalg.norm(updated) + 1e-6
            self.centroids[class_idx] = updated
            self.sample_counts[class_idx] += 1

    def clear(self) -> None:
        """Reset all learned centroids and counts."""
        self.centroids = [None] * self.num_classes
        self.sample_counts = [0] * self.num_classes

    def has_active_classes(self) -> bool:
        """Check if at least two classes have recorded samples."""
        active = sum(1 for c in self.centroids if c is not None)
        return active >= 2

    def predict(self, embedding: np.ndarray, temperature: float = 0.15) -> np.ndarray:
        """Compute Softmax probabilities against active class centroids."""
        logits = np.full(self.num_classes, -1.0, dtype=np.float32)

        for idx, centroid in enumerate(self.centroids):
            if centroid is not None:
                # Cosine similarity between unit vectors
                logits[idx] = float(np.dot(embedding, centroid))

        # Filter inactive classes by assigning large negative logit
        active_mask = np.array([c is not None for c in self.centroids])
        if not np.any(active_mask):
            return np.full(self.num_classes, 1.0 / self.num_classes, dtype=np.float32)

        logits[~active_mask] = -1e5
        return compute_softmax(logits, temperature)


def create_hud_header(
    width: int,
    fps: float,
    winning_class: str,
    winning_prob: float,
    temp: float,
    smoothing_enabled: bool,
    active_count: int,
) -> np.ndarray:
    """Render top telemetry header positioned strictly above the video."""
    hud_h = 90
    header = np.full((hud_h, width, 3), 20, dtype=np.uint8)

    status_str = f"PREDICTION: {winning_class} ({winning_prob * 100:.1f}%)"
    if active_count < 2:
        status_str = "CALIBRATION: Press [1], [2], [3], [4] to record samples"
        status_color = (0, 165, 255)
    else:
        status_color = (0, 255, 0)

    # Status row
    cv2.putText(
        header,
        f"{status_str} | FPS: {fps:.1f}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        1,
        cv2.LINE_AA,
    )

    # Telemetry metrics
    smooth_str = "ENABLED (alpha=0.20)" if smoothing_enabled else "OFF (Raw Softmax)"
    cv2.putText(
        header,
        f"Classes Active: {active_count}/4  |  Temperature: {temp:.2f} (+/-)  |  "
        f"Temporal EMA: {smooth_str} [t]",
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
        "[1-4] Add Sample  [r] Reset  [t] Toggle Smoothing  [c] Cam  [s] Snap  [q] Quit",
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


def render_probability_sidebar(
    height: int,
    probabilities: np.ndarray,
    class_names: list[str],
    sample_counts: list[int],
    winning_idx: int,
) -> np.ndarray:
    """Render a live probability bar chart panel."""
    panel_w = 280
    panel = np.full((height, panel_w, 3), 28, dtype=np.uint8)

    cv2.putText(
        panel,
        "TOP-K PROBABILITIES",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 200, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(panel, (15, 40), (panel_w - 15, 40), (70, 70, 70), 1)

    y_offset = 75
    bar_max_w = 170

    for idx, (name, prob, cnt) in enumerate(
        zip(class_names, probabilities, sample_counts, strict=False)
    ):
        is_winner = idx == winning_idx and cnt > 0
        text_color = (0, 255, 0) if is_winner else (200, 200, 200)
        bar_color = (0, 220, 0) if is_winner else (180, 120, 50)

        # Class label and sample count
        label = f"{name[:14]} (n={cnt})"
        cv2.putText(
            panel,
            label,
            (15, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            text_color,
            1,
            cv2.LINE_AA,
        )

        # Probability bar
        bar_w = int(prob * bar_max_w)
        cv2.rectangle(
            panel,
            (15, y_offset + 8),
            (15 + bar_max_w, y_offset + 22),
            (45, 45, 45),
            -1,
        )
        if bar_w > 0:
            cv2.rectangle(
                panel,
                (15, y_offset + 8),
                (15 + bar_w, y_offset + 22),
                bar_color,
                -1,
            )

        # Percentage text
        cv2.putText(
            panel,
            f"{prob * 100:.1f}%",
            (15 + bar_max_w + 10, y_offset + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            text_color,
            1,
            cv2.LINE_AA,
        )

        y_offset += 65

    # Instructions box at bottom
    cv2.putText(
        panel,
        "HOW TO CALIBRATE:",
        (15, height - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 165, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "1. Point camera at scene / object.",
        (15, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "2. Press [1], [2], [3] to train.",
        (15, height - 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    return panel


def main() -> None:
    """Main image classification and feature embedding loop."""
    parser = argparse.ArgumentParser(description="Module 10: Real-Time Image Classification")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--temp", type=float, default=0.15, help="Softmax temperature")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cam = ThreadedCamera(device_index=args.device, width=args.width, height=args.height).start()
    fps_meter = FPSMeter()
    classifier = EmbeddingClassifier(num_classes=4)

    temperature = args.temp
    smoothing_enabled = True
    smoothed_probs = np.full(4, 0.25, dtype=np.float32)
    ema_alpha = 0.20

    window_name = "Module 10: Image Classification & Feature Embeddings"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("=== Module 10: Image Classification Pipeline Initialized ===")
    print("Point camera at an object and press [1], [2], [3], [4] to record training samples.")

    try:
        while True:
            ret, frame, _ = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            h, w = frame.shape[:2]
            fps = fps_meter.tick()

            # Center crop region of interest for targeted classification
            crop_sz = min(w, h) // 2
            cx, cy = w // 2, h // 2
            x1, y1 = cx - crop_sz // 2, cy - crop_sz // 2
            x2, y2 = cx + crop_sz // 2, cy + crop_sz // 2
            focus_crop = frame[y1:y2, x1:x2]

            # Extract 128-D feature embedding from current focus crop
            embedding = extract_feature_embedding(focus_crop)

            # Predict Softmax probabilities
            raw_probs = classifier.predict(embedding, temperature)

            # Temporal Exponential Moving Average (EMA) smoothing
            if smoothing_enabled:
                smoothed_probs = ema_alpha * raw_probs + (1.0 - ema_alpha) * smoothed_probs
                active_probs = smoothed_probs
            else:
                active_probs = raw_probs

            # Determine winning prediction
            active_counts = sum(1 for c in classifier.centroids if c is not None)
            winning_idx = int(np.argmax(active_probs))
            winning_class = classifier.class_names[winning_idx]
            winning_prob = float(active_probs[winning_idx])

            # Draw visual focus reticle
            vis_video = frame.copy()
            reticle_color = (0, 255, 0) if active_counts >= 2 else (0, 165, 255)
            cv2.rectangle(vis_video, (x1, y1), (x2, y2), reticle_color, 2)

            # Corner brackets on focus box
            bracket_len = 15
            for bx, by in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                dx = bracket_len if bx == x1 else -bracket_len
                dy = bracket_len if by == y1 else -bracket_len
                cv2.line(vis_video, (bx, by), (bx + dx, by), reticle_color, 3)
                cv2.line(vis_video, (bx, by), (bx, by + dy), reticle_color, 3)

            # Focus label
            cv2.putText(
                vis_video,
                "CLASSIFICATION REGION (ROI)",
                (x1 + 6, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                reticle_color,
                1,
                cv2.LINE_AA,
            )

            # Render sidebar bar chart and concatenate with video canvas
            sidebar = render_probability_sidebar(
                h,
                active_probs,
                classifier.class_names,
                classifier.sample_counts,
                winning_idx,
            )
            video_with_sidebar = np.hstack([vis_video, sidebar])

            # Stack dedicated top telemetry header above the combined video
            hud = create_hud_header(
                video_with_sidebar.shape[1],
                fps,
                winning_class,
                winning_prob,
                temperature,
                smoothing_enabled,
                active_counts,
            )
            composite = np.vstack([hud, video_with_sidebar])

            cv2.imshow(window_name, composite)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                slot = int(chr(key)) - 1
                classifier.add_sample(slot, embedding)
                c_name = classifier.class_names[slot]
                c_cnt = classifier.sample_counts[slot]
                print(f"Added sample to {c_name} (n={c_cnt})")
            elif key == ord("t"):
                smoothing_enabled = not smoothing_enabled
                print(f"Temporal smoothing set to: {smoothing_enabled}")
            elif key == ord("r"):
                classifier.clear()
                smoothed_probs = np.full(4, 0.25, dtype=np.float32)
                print("All learned classes cleared.")
            elif key in (ord("+"), ord("=")):
                temperature = min(1.0, temperature + 0.02)
            elif key in (ord("-"), ord("_")):
                temperature = max(0.02, temperature - 0.02)
            elif key == ord("c"):
                next_device = 2 if cam.device_index == 0 else 0
                print(f"Switching camera to device index {next_device}...")
                cam.release()
                cam = ThreadedCamera(
                    device_index=next_device, width=args.width, height=args.height
                ).start()
            elif key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = output_dir / f"classification_{timestamp}.png"
                cv2.imwrite(str(save_path), composite)
                print(f"Snapshot saved to: {save_path.name}")

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
