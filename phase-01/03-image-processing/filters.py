"""Spatial filtering and convolution engines for visual computing.

Provides 2D kernel convolutions, smoothing, sharpening, bilateral filtering,
morphology, and masked portrait background blurring.
"""

from typing import Tuple

import cv2
import numpy as np
from scipy import signal


def create_gaussian_kernel_1d(ksize: int, sigma: float = 0.0) -> np.ndarray:
    """Generate a 1D Gaussian kernel using analytical math."""
    if ksize % 2 == 0:
        raise ValueError("Kernel size must be an odd positive integer.")
    if sigma <= 0.0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8

    radius = ksize // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def create_gaussian_kernel_2d(ksize: int, sigma: float = 0.0) -> np.ndarray:
    """Generate a 2D separable Gaussian kernel via outer product."""
    k1d = create_gaussian_kernel_1d(ksize, sigma)
    k2d = np.outer(k1d, k1d)
    return (k2d / k2d.sum()).astype(np.float32)


def custom_convolve2d_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Perform first-principles 2D spatial convolution on a single channel.

    Uses scipy.signal.convolve2d with boundary reflection for verification against OpenCV.
    """
    convolved = signal.convolve2d(channel, kernel, mode="same", boundary="symm")
    return np.clip(convolved, 0, 255).astype(np.uint8)


def apply_box_blur(frame: np.ndarray, ksize: int = 51) -> np.ndarray:
    """Apply uniform box blur averaging neighbors equally."""
    ksize = ksize if ksize % 2 != 0 else ksize + 1
    return cv2.blur(frame, (ksize, ksize))


def apply_gaussian_blur(frame: np.ndarray, ksize: int = 51, sigma: float = 0.0) -> np.ndarray:
    """Apply low-pass Gaussian smoothing with natural distance decay."""
    ksize = ksize if ksize % 2 != 0 else ksize + 1
    return cv2.GaussianBlur(frame, (ksize, ksize), sigma)


def apply_sharpen(frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Apply high-pass detail enhancement kernel."""
    # Base Laplacian sharpening kernel: center weight = 4 + strength
    kernel = np.array(
        [[0, -1, 0], [-1, 4.0 + strength, -1], [0, -1, 0]],
        dtype=np.float32,
    )
    return cv2.filter2D(frame, -1, kernel)


def apply_bilateral_filter(
    frame: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply edge-preserving smoothing.

    Smooths textures while keeping high-contrast edges sharp.
    """
    return cv2.bilateralFilter(frame, diameter, sigma_color, sigma_space)


def create_focal_mask(height: int, width: int, radius_scale: float = 0.45) -> np.ndarray:
    """Generate a smooth radial alpha mask centered in the frame.

    Mask has value 1.0 in the center subject zone and smoothly decays to 0.0.
    """
    cy, cx = height // 2, width // 2
    y, x = np.ogrid[:height, :width]

    # Normalized elliptical distance from center
    dist_sq = ((x - cx) / (cx * radius_scale)) ** 2 + ((y - cy) / (cy * radius_scale)) ** 2
    # Smooth cosine falloff
    dist = np.sqrt(dist_sq)
    mask = np.clip(1.0 - (dist - 0.7) / 0.6, 0.0, 1.0).astype(np.float32)

    # Blur mask boundary to ensure natural optical transition
    mask_blurred = cv2.GaussianBlur(mask, (31, 31), 0)
    return mask_blurred[:, :, None]


def apply_portrait_blur(frame: np.ndarray, blur_ksize: int = 35) -> np.ndarray:
    """Simulate Zoom/Meet portrait blur using alpha-masked composition.

    Blends a sharp center subject region with a heavily blurred background.
    Formula: Output = Mask * Sharp + (1 - Mask) * Blurred
    """
    h, w = frame.shape[:2]
    mask = create_focal_mask(h, w)
    blurred = apply_gaussian_blur(frame, ksize=blur_ksize)

    sharp_f = frame.astype(np.float32)
    blurred_f = blurred.astype(np.float32)

    composite = mask * sharp_f + (1.0 - mask) * blurred_f
    return np.clip(composite, 0, 255).astype(np.uint8)


def apply_morphological_gradient(frame: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Compute morphological gradient (dilation minus erosion) to isolate structural contours."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    return cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)


FILTER_NAMES = {
    1: "1. Passthrough (Raw)",
    2: "2. Box Blur (Averaging)",
    3: "3. Gaussian Blur (Low-pass)",
    4: "4. Sharpening (High-pass)",
    5: "5. Bilateral (Edge-preserving)",
    6: "6. Portrait Blur (Zoom/Meet Style)",
    7: "7. Morphological Gradient",
}


def process_frame(frame: np.ndarray, filter_mode: int, ksize: int = 51) -> Tuple[np.ndarray, str]:
    """Route a frame through the selected spatial filter mode."""
    mode_name = FILTER_NAMES.get(filter_mode, "Unknown")

    if filter_mode == 1:
        return frame, mode_name
    elif filter_mode == 2:
        return apply_box_blur(frame, ksize=ksize), f"Box Blur ({ksize}x{ksize})"
    elif filter_mode == 3:
        return apply_gaussian_blur(frame, ksize=ksize), f"Gaussian Blur ({ksize}x{ksize})"
    elif filter_mode == 4:
        return apply_sharpen(frame), mode_name
    elif filter_mode == 5:
        return apply_bilateral_filter(frame), mode_name
    elif filter_mode == 6:
        return apply_portrait_blur(frame, blur_ksize=ksize), f"Portrait Blur ({ksize}x{ksize})"
    elif filter_mode == 7:
        return apply_morphological_gradient(frame), mode_name

    return frame, mode_name
