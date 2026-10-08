"""Edge detection and spatial gradient operators.

Implements Sobel, Scharr, Laplacian, Canny edge detection,
and HSV gradient angle vector field visualization.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import cv2
import numpy as np


class EdgeMode(Enum):
    """Supported edge detection and gradient visualization modes."""

    CANNY = "Canny Pipeline"
    SOBEL = "Sobel (Gx + Gy)"
    SCHARR = "Scharr Derivative"
    LAPLACIAN = "Laplacian (2nd Deriv)"
    GRADIENT_ANGLE = "Gradient Orientation (HSV)"


@dataclass
class EdgeConfig:
    """Runtime configuration parameters for edge filters."""

    canny_lower: int = 50
    canny_upper: int = 150
    blur_kernel: int = 3  # Odd integer >= 1
    sobel_kernel: int = 3  # 1, 3, 5, or 7
    laplacian_kernel: int = 3  # 1, 3, 5, or 7


def compute_sobel(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Compute normalized first-order spatial gradient magnitude using Sobel kernels.

    Args:
        gray: Grayscale input frame (uint8).
        ksize: Aperture size for Sobel kernel (must be 1, 3, 5, or 7).

    Returns:
        Gradient magnitude image normalized to uint8 [0, 255].
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.hypot(gx, gy)
    return np.uint8(np.clip(mag / (mag.max() + 1e-6) * 255.0, 0, 255))


def compute_scharr(gray: np.ndarray) -> np.ndarray:
    """Compute rotational-invariant first-order gradient magnitude via Scharr.

    Args:
        gray: Grayscale input frame (uint8).

    Returns:
        Normalized gradient magnitude image (uint8).
    """
    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    mag = np.hypot(gx, gy)
    return np.uint8(np.clip(mag / (mag.max() + 1e-6) * 255.0, 0, 255))


def compute_laplacian(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Compute second-order isotropic spatial derivative using Laplacian.

    Args:
        gray: Grayscale input frame (uint8).
        ksize: Aperture size (must be 1, 3, 5, or 7).

    Returns:
        Absolute Laplacian response image normalized to uint8.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    lap_abs = np.abs(lap)
    return np.uint8(np.clip(lap_abs / (lap_abs.max() + 1e-6) * 255.0, 0, 255))


def compute_canny(gray: np.ndarray, lower: int, upper: int) -> np.ndarray:
    """Compute binary single-pixel edges via the multi-stage Canny algorithm.

    Args:
        gray: Pre-smoothed grayscale input frame (uint8).
        lower: Lower hysteresis threshold.
        upper: Upper hysteresis threshold.

    Returns:
        Binary edge mask (uint8).
    """
    return cv2.Canny(gray, lower, upper)


def compute_gradient_angle_map(
    gray: np.ndarray,
    ksize: int = 3,
    magnitude_threshold: float = 30.0,
) -> np.ndarray:
    """Encode 2D gradient orientation as color hues in an HSV vector field.

    - Hue: Gradient direction theta = atan2(gy, gx) mapped from [-pi, pi] to [0, 179].
    - Saturation: 255 (maximum color purity).
    - Value: Gradient magnitude normalized and masked to reject sensor noise.

    Args:
        gray: Grayscale input frame (uint8).
        ksize: Kernel aperture for derivative calculation.
        magnitude_threshold: Gradients weaker than this are suppressed to black.

    Returns:
        BGR visualization frame (uint8).
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # In OpenCV HSV, Hue is [0, 179] (representing 0 to 360 degrees / 2)
    h = (angle / 2.0).astype(np.uint8)
    s = np.full_like(h, 255)

    # Normalize magnitude to [0, 255] and clamp noise
    mag_normalized = np.clip((mag / (mag.max() + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    v = np.where(mag > magnitude_threshold, mag_normalized, 0).astype(np.uint8)

    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def process_edges(
    frame: np.ndarray,
    mode: EdgeMode,
    config: EdgeConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute active edge operator and return pair of (annotated_input, edge_visualization).

    Args:
        frame: Input BGR frame.
        mode: Active edge detection mode.
        config: Edge detection thresholds and filter settings.

    Returns:
        Tuple of (annotated_input_frame, processed_edge_output).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pre-smooth to suppress high-frequency sensor noise
    k = max(1, config.blur_kernel | 1)  # Ensure odd kernel >= 1
    if k > 1:
        smoothed = cv2.GaussianBlur(gray, (k, k), sigmaX=0)
    else:
        smoothed = gray

    if mode == EdgeMode.CANNY:
        processed = compute_canny(smoothed, config.canny_lower, config.canny_upper)
        # Convert single-channel binary edges to BGR for consistent display
        vis = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    elif mode == EdgeMode.SOBEL:
        processed = compute_sobel(smoothed, ksize=config.sobel_kernel)
        vis = cv2.applyColorMap(processed, cv2.COLORMAP_VIRIDIS)
    elif mode == EdgeMode.SCHARR:
        processed = compute_scharr(smoothed)
        vis = cv2.applyColorMap(processed, cv2.COLORMAP_PLASMA)
    elif mode == EdgeMode.LAPLACIAN:
        processed = compute_laplacian(smoothed, ksize=config.laplacian_kernel)
        vis = cv2.applyColorMap(processed, cv2.COLORMAP_INFERNO)
    elif mode == EdgeMode.GRADIENT_ANGLE:
        vis = compute_gradient_angle_map(smoothed, ksize=config.sobel_kernel)
    else:
        vis = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)

    return frame.copy(), vis


def create_split_view(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    left_title: str = "Live Feed",
    right_title: str = "Edge Pipeline",
) -> np.ndarray:
    """Render side-by-side split screen for interactive comparison."""
    h, w = left_frame.shape[:2]
    half_w = w // 2
    resized_left = cv2.resize(left_frame, (half_w, h // 2))
    resized_right = cv2.resize(right_frame, (half_w, h // 2))

    # Add pane titles
    cv2.putText(resized_left, left_title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(
        resized_right, right_title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
    )

    return np.hstack((resized_left, resized_right))
