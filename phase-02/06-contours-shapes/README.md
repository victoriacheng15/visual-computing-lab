# Module 06: Contours, Shape Analysis & Geometry

## Overview

In Module 05, we used edge operators to detect pixel-level gradient discontinuities. However, raw edge pixels are disconnected coordinates: the computer does not know which pixels belong to the same boundary, what shape they enclose, or how that shape is oriented.

This module introduces topological boundary extraction, contour hierarchy trees, polygon approximation, and geometric invariants. We transition from low-level pixel manipulation to structural shape analysis:
1. Border-following topological extraction via Suzuki's algorithm (`cv2.findContours`).
2. Hierarchy trees (`RETR_TREE`, `RETR_CCOMP`) encoding parent-child hole relationships.
3. Spatial, central, and scale-invariant Hu Moments for shape descriptors.
4. Polygon approximation via the Douglas-Peucker algorithm (`cv2.approxPolyDP`).
5. Convex hulls, convexity defects, and minimum enclosing geometry (oriented bounding box with rotation angle $\theta$, minimum enclosing circle, fitted ellipse).
6. Real-time geometric shape classifier recognizing triangles, rectangles, squares, circles, and polygons with orientation angles.

---

## Key Engineering Concepts

### 1. Border Following & Topological Extraction
OpenCV implements the topological border-following algorithm formulated by Satoshi Suzuki and Keiichi Abe (1985). Given a binary mask, the algorithm traces connected components of boundary pixels using 8-connectivity.

Contour retrieval modes govern topology representation:
- **`RETR_EXTERNAL`:** Retrieves only outermost boundaries, ignoring all interior holes. Fastest when nested geometry is irrelevant.
- **`RETR_TREE`:** Reconstructs the complete nested hierarchy of parent outlines and nested child holes as a 4-element integer array per contour `[Next, Previous, First_Child, Parent]`.

### 2. Douglas-Peucker Polygon Approximation
To classify geometric shapes, raw irregular contour boundaries are simplified into polygonal vertices using the Douglas-Peucker algorithm (`cv2.approxPolyDP`):

$$\epsilon = \alpha \times \text{Perimeter}(C)$$

Where $\epsilon$ is the maximum perpendicular distance a vertex can deviate from the approximating line segment.
- Vertex count $V = 3$: Triangle.
- Vertex count $V = 4$: Quadrilateral. Evaluated via aspect ratio $\frac{W}{H}$ to distinguish Square ($0.92 \le \frac{W}{H} \le 1.08$) from Rectangle.
- Vertex count $V = 5$: Pentagon.
- Vertex count $V > 6$: Circularity metric determines Circle vs. generic polygon.

### 3. Circularity and Compactness
A circle minimizes perimeter for a given enclosed area. The isoperimetric quotient (circularity metric) is defined as:

$$\mathcal{C} = \frac{4\pi \times \text{Area}}{\text{Perimeter}^2}$$

- Perfect circle: $\mathcal{C} \approx 1.0$.
- Square: $\mathcal{C} = \frac{\pi}{4} \approx 0.785$.
- Equilateral triangle: $\mathcal{C} = \frac{\pi}{3\sqrt{3}} \approx 0.604$.
- Thin, elongated shapes: $\mathcal{C} \to 0.0$.

### 4. Oriented Bounding Geometry & Rotation Angle
Axis-aligned bounding boxes (`cv2.boundingRect`) change dimensions when an object rotates. To extract invariant physical dimensions, we fit minimum-area oriented bounding boxes:
- **`cv2.minAreaRect`:** Computes the smallest rotated rectangle enclosing the contour, returning center $(x, y)$, dimensions $(w, h)$, and rotation angle $\theta \in [-90^\circ, 0^\circ]$.
- **`cv2.fitEllipse`:** Fits an ellipse via algebraic least squares, deriving major/minor axis lengths and orientation angle.

### 5. Convex Hull & Convexity Defects
- **Convex Hull (`cv2.convexHull`):** The smallest convex polygon enclosing all contour points (computed using Sklansky's algorithm).
- **Convexity Defects (`cv2.convexityDefects`):** The cavities or valleys where the contour deviates inward from the convex hull. By measuring defect depth and angles, vision systems count extended fingers or distinguish complex concave objects.

---

## File Structure

```text
phase-02/06-contours-shapes/
├── README.md
├── detector.py    # Shape classification engine, polygon approx, and geometric fitting
└── main.py        # Real-time interactive shape lab with split-screen HUD & trackbars
```

---

## Running the Module

Execute via the project Makefile:

```bash
make run phase-02/06-contours-shapes/main.py
```

Or directly via `uv`:

```bash
uv run python phase-02/06-contours-shapes/main.py
```

### Interactive Controls
- **`1`:** Shape Classifier Mode (labels triangles, rectangles, squares, pentagons, circles with angles)
- **`2`:** Oriented Bounding Box Mode (`minAreaRect` with orientation angle $\theta$)
- **`3`:** Convex Hull & Defects Mode (enclosing hull and interior defect valleys)
- **`4`:** Minimum Enclosing Circle & Ellipse Mode
- **`5`:** Raw Contour Hierarchy Mode (visualizes parent outlines and child holes)
- **`Sliders`:** Adjust binary threshold level and epsilon approximation factor ($\alpha$)
- **`c`:** Cycle camera device (0 <-> 2)
- **`s`:** Save snapshot and dump detected shape geometry to `output/`
- **`q` or `Esc`:** Quit
