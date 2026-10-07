# Module 03: Image Processing & Spatial Filtering

## Overview

In digital signal processing, an image is filtered by convolving a 2D matrix of pixel intensities with a smaller mathematical matrix called a **kernel** (or point spread function).

This module establishes the foundational mechanics of spatial convolution, noise reduction, detail sharpening, edge-preserving smoothing, and alpha-blended portrait blur (the mathematical engine behind Zoom and Google Meet background blur).

---

## Key Engineering Concepts

### 1. 2D Discrete Spatial Convolution
Convolution slides an odd-sized kernel $K$ of dimensions $(2k+1) \times (2k+1)$ over every pixel $(x, y)$ of the source image $I$:

$$(I * K)(x, y) = \sum_{i=-k}^{k} \sum_{j=-k}^{k} I(x - i, y - j) \cdot K(i, j)$$

- **Anchor Point:** Typically the exact center of the kernel.
- **Border Handling:** At image borders, out-of-bounds pixels are handled via border extrapolation (e.g., `BORDER_REFLECT_101` or `BORDER_CONSTANT`).
- **Normalized Kernels:** Blur kernels sum to $1.0$ to preserve total scene energy and brightness.

### 2. Common Spatial Kernels

#### Box Blur (Uniform Average)
A square matrix where every element is identical: $K_{i,j} = \frac{1}{N^2}$. Fast, but produces boxy artifacts because all neighbors contribute equally regardless of distance.

#### Gaussian Blur (Low-Pass Filter)
Weights pixels by a 2D Gaussian bell curve based on radial distance from the center pixel:

$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

Closer pixels contribute strongly; distant pixels contribute weakly. This produces natural optical softness without ringing artifacts.

#### Sharpening Kernel (High-Pass Amplification)
Enhances high-frequency spatial gradients by amplifying the center pixel while subtracting the surrounding average:

$$K_{\text{sharpen}} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$

### 3. Bilateral Filter (Edge-Preserving Smoothing)
Standard Gaussian blur smooths across edges, blurring sharp object boundaries. A **bilateral filter** considers two weights for each neighbor:
1. **Spatial distance:** How close is the pixel in coordinate space $(x, y)$?
2. **Photometric distance:** How close is the pixel in intensity/color similarity $|I_p - I_q|$?

If a neighboring pixel has a drastically different color (indicating a sharp edge), its weight drops to near zero. Result: skin surfaces and textures are smoothed, while sharp contours and eyes remain crisp.

### 4. Alpha Blending & Portrait Blur (Zoom / Meet Engine)
Background blur combines two frame buffers using a spatial weight mask $M(x, y) \in [0.0, 1.0]$:

$$I_{\text{composite}} = M \cdot I_{\text{sharp}} + (1 - M) \cdot I_{\text{blurred}}$$

In this module, $M$ is generated as a soft geometric focal mask. In Module 11, $M$ is replaced with a live neural person segmentation mask to build a complete virtual background pipeline.

---

## File Structure

```text
phase-01/03-image-processing/
├── README.md
├── filters.py    # Vectorized convolution and spatial filter implementations
└── main.py       # Live camera stream with interactive filter switcher
```

---

## Running the Module

Execute via the project Makefile:

```bash
make run phase-01/03-image-processing/main.py
```

Or directly via `uv`:

```bash
uv run python phase-01/03-image-processing/main.py
```

### Interactive Filter Controls
- **`1`:** Passthrough (Normal raw feed)
- **`2`:** Box Blur (Mean averaging)
- **`3`:** Gaussian Blur (Smooth low-pass)
- **`4`:** Sharpening (Detail enhancement)
- **`5`:** Bilateral Filter (Edge-preserving skin smoothing)
- **`6`:** Portrait Blur (Focal mask sharp subject + blurred background)
- **`7`:** Morphological Gradient (Outer contour skeleton)
- **`+` / `-`:** Dynamically increase or decrease blur kernel size ($3 \times 3$ up to $151 \times 151$)
- **`c`:** Switch active camera (0 <-> 2)
- **`s`:** Save snapshot to `phase-01/03-image-processing/output/`
- **`q` or `Esc`:** Quit
