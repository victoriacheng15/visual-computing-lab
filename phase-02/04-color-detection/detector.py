"""Color detection and spatial segmentation engine in HSV color space.

Handles cylindrical Hue wrap-around, morphological noise suppression,
and centroid extraction via spatial moments.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class HSVRange:
    """Represents bounds for color segmentation in OpenCV HSV space."""

    name: str
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int
    wrap_red: bool = False

    def to_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert standard bounds to NumPy uint8 arrays."""
        lower = np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)
        upper = np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)
        return lower, upper


# Tuned baseline color presets for standard indoor lighting
COLOR_PRESETS: dict[int, HSVRange] = {
    1: HSVRange("Red", h_min=0, h_max=10, s_min=100, s_max=255, v_min=70, v_max=255, wrap_red=True),
    2: HSVRange("Green", h_min=35, h_max=85, s_min=80, s_max=255, v_min=50, v_max=255),
    3: HSVRange("Blue", h_min=95, h_max=130, s_min=100, s_max=255, v_min=60, v_max=255),
    4: HSVRange("Yellow", h_min=20, h_max=35, s_min=100, s_max=255, v_min=100, v_max=255),
}


def build_color_mask(hsv_frame: np.ndarray, hsv_range: HSVRange) -> np.ndarray:
    """Generate a clean binary mask isolating pixels within the target HSV range.

    Handles Red hue wrap-around at 0/180 degrees and cleans noise via morphology.
    """
    if hsv_range.wrap_red:
        # Lower red band: [0, s_min, v_min] to [10, s_max, v_max]
        lower1 = np.array([0, hsv_range.s_min, hsv_range.v_min], dtype=np.uint8)
        upper1 = np.array([hsv_range.h_max, hsv_range.s_max, hsv_range.v_max], dtype=np.uint8)
        mask1 = cv2.inRange(hsv_frame, lower1, upper1)

        # Upper red band: [170, s_min, v_min] to [179, s_max, v_max]
        lower2 = np.array([170, hsv_range.s_min, hsv_range.v_min], dtype=np.uint8)
        upper2 = np.array([179, hsv_range.s_max, hsv_range.v_max], dtype=np.uint8)
        mask2 = cv2.inRange(hsv_frame, lower2, upper2)

        raw_mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower, upper = hsv_range.to_bounds()
        raw_mask = cv2.inRange(hsv_frame, lower, upper)

    # Morphological noise filtering:
    # 1. Opening (erode then dilate) removes salt-and-pepper white specks
    # 2. Closing (dilate then erode) fills small holes inside detected regions
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

    return clean_mask


def extract_target_geometry(
    binary_mask: np.ndarray,
    min_area: float = 600.0,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int, int, int]], float]:
    """Calculate centroid and bounding box from the largest blob in the binary mask.

    Returns:
        Tuple of ((cx, cy), (x, y, w, h), area). Returns (None, None, 0.0) if no target found.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0

    # Locate contour with largest area
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < min_area:
        return None, None, area

    # Compute center of mass via spatial moments
    moments = cv2.moments(largest_contour)
    if moments["m00"] <= 0.0:
        return None, None, area

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    bbox = cv2.boundingRect(largest_contour)
    return (cx, cy), bbox, area


def annotate_frame(
    frame: np.ndarray,
    centroid: Optional[Tuple[int, int]],
    bbox: Optional[Tuple[int, int, int, int]],
    color_name: str,
    area: float,
) -> np.ndarray:
    """Draw bounding box, center crosshair, and telemetry on the frame."""
    annotated = frame.copy()

    if bbox is not None and centroid is not None:
        x, y, w, h = bbox
        cx, cy = centroid

        # Bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Centroid target
        cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)
        cv2.circle(annotated, (cx, cy), 16, (0, 255, 255), 1)

        # Coordinate label
        label = f"{color_name}: ({cx}, {cy}) | {int(area)}px"
        cv2.putText(
            annotated,
            label,
            (x, max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


def create_split_view(annotated_frame: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """Combine RGB tracking frame and binary detection mask side-by-side."""
    h, w = annotated_frame.shape[:2]
    # Downsample width by half to create a clean 16:9 side-by-side window
    half_w = w // 2
    resized_rgb = cv2.resize(annotated_frame, (half_w, h // 2))

    mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    resized_mask = cv2.resize(mask_bgr, (half_w, h // 2))

    # Add pane titles
    cv2.putText(
        resized_rgb, "Tracked Object", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
    )
    cv2.putText(
        resized_mask, "Binary HSV Mask", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1
    )

    return np.hstack((resized_rgb, resized_mask))
