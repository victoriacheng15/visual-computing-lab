"""Contour analysis, geometric shape classification, and boundary fitting engine.

Provides algorithms for polygon approximation, circularity estimation,
minimum-area oriented bounding boxes, convex hulls, and shape labeling.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np


class ShapeMode(Enum):
    """Visualization and analysis modes for contour geometries."""

    CLASSIFIER = "Geometric Shape Classifier"
    ORIENTED_BBOX = "Oriented Bounding Box (minAreaRect)"
    CONVEX_HULL = "Convex Hull & Defects"
    ENCLOSING_CIRCLE = "Enclosing Circle & Ellipse"
    HIERARCHY = "Contour Tree Hierarchy"


@dataclass
class ShapeConfig:
    """Runtime parameters for contour extraction and approximation."""

    binary_thresh: int = 120
    epsilon_percent: int = 3  # Epsilon factor alpha = percent / 100.0 (e.g. 0.03)
    min_area: float = 800.0
    invert_binary: bool = False


@dataclass
class DetectedShape:
    """Represents a classified geometric shape and its properties."""

    name: str
    centroid: Tuple[int, int]
    area: float
    perimeter: float
    circularity: float
    vertex_count: int
    contour: np.ndarray
    approx_poly: np.ndarray
    oriented_box: Optional[np.ndarray] = None
    orientation_deg: float = 0.0


def extract_binary_mask(
    gray: np.ndarray,
    thresh_val: int,
    invert: bool = False,
) -> np.ndarray:
    """Generate a clean binary threshold mask with Otsu or fixed threshold.

    Pre-smooths with Gaussian blur to prevent jagged boundary artifacts.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blurred, thresh_val, 255, thresh_type)

    # Clean morphological pinholes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def classify_single_contour(
    contour: np.ndarray,
    epsilon_alpha: float = 0.03,
) -> Optional[DetectedShape]:
    """Analyze contour geometry and classify into geometric primitives.

    Args:
        contour: 2D contour point array.
        epsilon_alpha: Douglas-Peucker approximation factor relative to perimeter.

    Returns:
        DetectedShape metadata or None if contour is degenerate.
    """
    area = cv2.contourArea(contour)
    if area < 10.0:
        return None

    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter <= 0.0:
        return None

    # Calculate spatial moments for centroid
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])

    # Circularity: (4 * pi * Area) / (Perimeter^2)
    circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

    # Douglas-Peucker polygon approximation
    epsilon = epsilon_alpha * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    vertex_count = len(approx)

    # Oriented bounding box
    rect = cv2.minAreaRect(contour)  # ((center_x, center_y), (w, h), angle)
    box_points = cv2.boxPoints(rect).astype(np.int32)
    (rect_w, rect_h) = rect[1]
    angle = rect[2]

    # Normalize orientation angle
    if rect_w < rect_h:
        angle = angle + 90.0

    # Classification heuristics based on vertices, circularity, and aspect ratio
    shape_name = "Polygon"

    if vertex_count == 3:
        shape_name = "Triangle"
    elif vertex_count == 4:
        # Check aspect ratio
        aspect_ratio = float(rect_w) / max(1.0, float(rect_h))
        if 0.88 <= aspect_ratio <= 1.14:
            shape_name = "Square"
        else:
            shape_name = "Rectangle"
    elif vertex_count == 5:
        shape_name = "Pentagon"
    elif vertex_count == 6:
        shape_name = "Hexagon"
    else:
        # High vertex count: check if circular
        if circularity >= 0.78:
            shape_name = "Circle"
        else:
            shape_name = f"Polygon ({vertex_count}v)"

    return DetectedShape(
        name=shape_name,
        centroid=(cx, cy),
        area=area,
        perimeter=perimeter,
        circularity=circularity,
        vertex_count=vertex_count,
        contour=contour,
        approx_poly=approx,
        oriented_box=box_points,
        orientation_deg=angle,
    )


def render_shape_overlays(
    canvas: np.ndarray,
    shapes: List[DetectedShape],
    mode: ShapeMode,
    contours: List[np.ndarray],
    hierarchy: Optional[np.ndarray],
) -> np.ndarray:
    """Render analytical geometric overlays based on active mode."""
    overlay = canvas.copy()

    if mode == ShapeMode.HIERARCHY and hierarchy is not None:
        # Draw full contour tree with hierarchy colors
        for i, cnt in enumerate(contours):
            # Parent vs child color code
            is_child = hierarchy[0][i][3] != -1
            color = (0, 165, 255) if is_child else (0, 255, 0)
            cv2.drawContours(overlay, [cnt], -1, color, 2)
        return overlay

    for s in shapes:
        cx, cy = s.centroid

        if mode == ShapeMode.CLASSIFIER:
            # Draw approximated polygon and vertices
            cv2.drawContours(overlay, [s.approx_poly], -1, (0, 255, 255), 2)
            for pt in s.approx_poly:
                cv2.circle(overlay, tuple(pt[0]), 4, (0, 0, 255), -1)

            # Draw centroid and label
            cv2.circle(overlay, (cx, cy), 5, (255, 0, 0), -1)
            label = f"{s.name} | {s.orientation_deg:3.0f} deg"
            cv2.putText(
                overlay,
                label,
                (cx - 30, max(24, cy - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        elif mode == ShapeMode.ORIENTED_BBOX:
            # Draw minimum-area rotated bounding box
            if s.oriented_box is not None:
                cv2.drawContours(overlay, [s.oriented_box], -1, (255, 120, 0), 2)

            # Draw principal orientation axis
            rad = np.deg2rad(s.orientation_deg)
            length = 40
            x2 = int(cx + length * np.cos(rad))
            y2 = int(cy + length * np.sin(rad))
            cv2.arrowedLine(overlay, (cx, cy), (x2, y2), (0, 255, 255), 2, tipLength=0.3)

            label = f"{s.name}: {s.orientation_deg:4.1f} deg"
            cv2.putText(
                overlay,
                label,
                (cx - 20, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        elif mode == ShapeMode.CONVEX_HULL:
            # Draw raw contour in green
            cv2.drawContours(overlay, [s.contour], -1, (0, 255, 0), 1)

            # Draw convex hull in magenta
            hull = cv2.convexHull(s.contour)
            cv2.drawContours(overlay, [hull], -1, (255, 0, 255), 2)

            # Compute convexity defects if contour is non-trivial
            hull_indices = cv2.convexHull(s.contour, returnPoints=False)
            if hull_indices is not None and len(hull_indices) > 3 and len(s.contour) > 3:
                try:
                    defects = cv2.convexityDefects(s.contour, hull_indices)
                    if defects is not None:
                        # Flatten defect entries safely across OpenCV versions
                        defect_rows = defects.reshape(-1, 4)
                        for s_idx, e_idx, f_idx, dist in defect_rows:
                            # Only draw significant defect valleys (distance > 10px)
                            if dist > 10 * 256:
                                far = tuple(s.contour[f_idx][0])
                                cv2.circle(overlay, far, 5, (0, 0, 255), -1)
                except cv2.error:
                    pass

        elif mode == ShapeMode.ENCLOSING_CIRCLE:
            # Minimum enclosing circle
            (circ_x, circ_y), radius = cv2.minEnclosingCircle(s.contour)
            cv2.circle(
                overlay,
                (int(circ_x), int(circ_y)),
                int(radius),
                (0, 255, 255),
                2,
            )

            # Fit ellipse if enough points exist
            if len(s.contour) >= 5:
                ellipse = cv2.fitEllipse(s.contour)
                cv2.ellipse(overlay, ellipse, (255, 0, 128), 2)

            cv2.putText(
                overlay,
                f"Circularity: {s.circularity:.2f}",
                (cx - 40, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return overlay


def process_shapes(
    frame: np.ndarray,
    mode: ShapeMode,
    config: ShapeConfig,
) -> Tuple[np.ndarray, np.ndarray, List[DetectedShape]]:
    """Execute complete shape extraction, approximation, and classification pipeline."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = extract_binary_mask(gray, config.binary_thresh, config.invert_binary)

    # Topological border following
    retrieval_mode = cv2.RETR_TREE if mode == ShapeMode.HIERARCHY else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(mask, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)

    epsilon_alpha = max(0.005, config.epsilon_percent / 100.0)
    shapes: List[DetectedShape] = []

    for cnt in contours:
        if cv2.contourArea(cnt) < config.min_area:
            continue
        shape_meta = classify_single_contour(cnt, epsilon_alpha)
        if shape_meta is not None:
            shapes.append(shape_meta)

    # Sort shapes by descending area
    shapes.sort(key=lambda s: s.area, reverse=True)

    # Generate analytical overlay
    overlay = render_shape_overlays(frame, shapes, mode, contours, hierarchy)

    # Composite mask representation (3-channel for side-by-side view)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return overlay, mask_bgr, shapes


def create_split_view(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    left_title: str = "Shape Analysis Overlay",
    right_title: str = "Binary Segmentation Mask",
) -> np.ndarray:
    """Render side-by-side split screen for interactive feedback."""
    h, w = left_frame.shape[:2]
    half_w = w // 2
    resized_left = cv2.resize(left_frame, (half_w, h // 2))
    resized_right = cv2.resize(right_frame, (half_w, h // 2))

    cv2.putText(resized_left, left_title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(
        resized_right, right_title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
    )

    return np.hstack((resized_left, resized_right))
