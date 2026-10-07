"""Module 01: Image Basics.

Demonstrates foundational digital image data structures, memory layout,
coordinate systems, channel slicing, and bit-depth normalization using NumPy and OpenCV.
"""

import os
from pathlib import Path

# Fall back gracefully to XWayland (xcb) if native Wayland plugin cannot initialize
if os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland")

import cv2
import numpy as np


def create_synthetic_color_bars(height: int = 360, width: int = 640) -> np.ndarray:
    """Create a synthetic 8-color test card in BGR uint8 format.
    Colors: White, Yellow, Cyan, Green, Magenta, Red, Blue, Black.
    """
    # 8 standard SMPTE-like test colors in BGR order
    bgr_palette = [
        (255, 255, 255),  # White
        (0, 255, 255),  # Yellow
        (255, 255, 0),  # Cyan
        (0, 255, 0),  # Green
        (255, 0, 255),  # Magenta
        (0, 0, 255),  # Red
        (255, 0, 0),  # Blue
        (0, 0, 0),  # Black
    ]

    image = np.zeros((height, width, 3), dtype=np.uint8)
    num_bars = len(bgr_palette)
    bar_width = width // num_bars

    for i, color in enumerate(bgr_palette):
        x_start = i * bar_width
        x_end = width if i == num_bars - 1 else (i + 1) * bar_width
        image[:, x_start:x_end] = color

    return image


def create_gradient_ramp(height: int = 360, width: int = 640) -> np.ndarray:
    """Create a horizontal luminance gradient ramp in float32 format [0.0, 1.0]."""
    # 1D linear ramp along width, broadcast across height
    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
    gradient = np.tile(ramp, (height, 1))
    return gradient


def inspect_array_metadata(name: str, array: np.ndarray) -> None:
    """Print critical memory layout details of a NumPy array."""
    print(f"\n--- Metadata: {name} ---")
    print(f"  Shape:       {array.shape} (Height, Width, Channels)")
    print(f"  Dtype:       {array.dtype}")
    print(f"  Item size:   {array.itemsize} bytes")
    print(f"  Total bytes: {array.nbytes} bytes")
    print(f"  Strides:     {array.strides}")
    print(f"  Contiguous:  {array.flags['C_CONTIGUOUS']}")


def main() -> None:
    print("========================================")
    print(" Visual Computing Lab - Module 01")
    print(" First-Principles Image Array Operations")
    print("========================================")

    # 1. Generate synthetic color bars
    color_bars = create_synthetic_color_bars()
    inspect_array_metadata("Synthetic Color Bars (uint8 BGR)", color_bars)

    # 2. Vectorized channel slicing (no loops)
    # OpenCV BGR layout: index 0 is Blue, 1 is Green, 2 is Red
    blue_channel = color_bars[:, :, 0]
    green_channel = color_bars[:, :, 1]
    red_channel = color_bars[:, :, 2]
    b_mean, g_mean, r_mean = blue_channel.mean(), green_channel.mean(), red_channel.mean()
    print(f"\nChannel Means -> B: {b_mean:.1f}, G: {g_mean:.1f}, R: {r_mean:.1f}")

    # 3. Channel reversal: BGR to RGB via negative slicing
    # Note: strided slicing produces a non-contiguous memory view
    rgb_view = color_bars[:, :, ::-1]
    inspect_array_metadata("RGB Strided Slice (non-contiguous)", rgb_view)

    # Enforce C-contiguity for C-extension safety
    rgb_contiguous = np.ascontiguousarray(rgb_view)
    inspect_array_metadata("RGB Contiguous Array", rgb_contiguous)

    # 4. Region of Interest (ROI) slicing and modification
    # Coordinates in array indexing: [row_start:row_end, col_start:col_end] -> [y1:y2, x1:x2]
    canvas = color_bars.copy()
    h, w, _ = canvas.shape
    roi_h, roi_w = 80, 120
    y1, y2 = (h - roi_h) // 2, (h + roi_h) // 2
    x1, x2 = (w - roi_w) // 2, (w + roi_w) // 2

    # Draw an inverted patch in the center
    canvas[y1:y2, x1:x2] = 255 - canvas[y1:y2, x1:x2]

    # 5. Bit-depth conversion & Normalization
    gradient = create_gradient_ramp()
    inspect_array_metadata("Luminance Gradient (float32 [0.0, 1.0])", gradient)
    # Convert float32 back to displayable uint8 [0, 255]
    gradient_uint8 = (gradient * 255.0).astype(np.uint8)

    # 6. Save outputs to disk
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    out_bars_path = output_dir / "color_bars_roi.png"
    out_grad_path = output_dir / "gradient_ramp.png"

    cv2.imwrite(str(out_bars_path), canvas)
    cv2.imwrite(str(out_grad_path), gradient_uint8)
    print(f"\n[Saved] Output saved to: {out_bars_path}")
    print(f"[Saved] Output saved to: {out_grad_path}")

    # 7. Interactive window if display server is present
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if has_display:
        print("\nDisplay server detected. Opening interactive window (press any key to exit)...")
        cv2.imshow("Module 01: Color Bars with Inverted Center ROI", canvas)
        cv2.imshow("Module 01: Gradient Ramp", gradient_uint8)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\nHeadless environment detected. Skipping cv2.imshow().")


if __name__ == "__main__":
    main()
