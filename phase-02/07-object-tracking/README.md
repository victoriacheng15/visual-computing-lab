# Module 07: Object Tracking & State Estimation

## Overview

In Modules 04–06, detection algorithms operated independently on single frames without temporal continuity: if a target passed behind an obstacle for a fraction of a second, the detector lost it completely.

This module introduces temporal tracking and state estimation. We transition from stateless per-frame detection to continuous trajectory modeling across time:
1. Centroid tracking with Euclidean distance bipartite association.
2. Sparse optical flow using the Lucas-Kanade differential method (`cv2.calcOpticalFlowPyrLK`).
3. Dense optical flow using Gunnar Farnebäck's polynomial expansion algorithm (`cv2.calcOpticalFlowFarneback`).
4. Color-histogram tracking via CamShift (Continuously Adaptive Mean-Shift).
5. State estimation using a 2D Linear Kalman Filter with constant velocity physics ($\mathbf{x} = [x, y, v_x, v_y]^T$) for trajectory smoothing and occlusion bridging.

---

## Key Engineering Concepts

### 1. Centroid Tracking & ID Assignment
Centroid tracking maintains object identities across successive frames ($t-1 \to t$).
- Compute the Euclidean distance matrix between existing target centroids and newly detected candidate centroids.
- Solve the assignment problem to associate detections to existing tracks.
- Increment consecutive lost-frame counters for unmatched tracks, purging trajectories that exceed a maximum staleness threshold.

### 2. Optical Flow (Lucas-Kanade & Farnebäck)
Optical flow calculates the apparent motion of image intensity patterns between consecutive frames under the brightness constancy assumption ($I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t)$):

$$I_x u + I_y v + I_t = 0$$

- **Lucas-Kanade (Sparse):** Solves for motion vectors $(u, v)$ over a local $N \times N$ neighborhood around high-contrast corner points (Shi-Tomasi corners) using image pyramids for multi-scale displacement.
- **Farnebäck (Dense):** Approximates neighborhoods of both frames using quadratic polynomials, producing a dense motion vector field across every pixel.

### 3. CamShift (Continuously Adaptive Mean-Shift)
CamShift adapts the Mean-Shift algorithm to track non-rigid, scaling targets:
1. Converts the target bounding box to an HSV color histogram (Hue channel backprojection).
2. Calculates the zeroth ($M_{00}$) and first-order ($M_{10}, M_{01}$) spatial moments of the probability distribution.
3. Continuously shifts the search window center toward the probability centroid while dynamically updating window size and orientation angle.

### 4. Kalman Filter State Estimation
A linear discrete Kalman filter estimates the true kinematic state of an object in the presence of sensor noise and temporary occlusions:

$$\mathbf{x}_k = \mathbf{F} \mathbf{x}_{k-1} + \mathbf{w}_{k-1}, \quad \mathbf{z}_k = \mathbf{H} \mathbf{x}_k + \mathbf{v}_k$$

Where the state vector models 2D position and velocity:

$$\mathbf{x} = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}, \quad \mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

- **Predict Step:** Projects state $\hat{\mathbf{x}}_k^-$ and error covariance $\mathbf{P}_k^-$ forward using the physical transition matrix $\mathbf{F}$.
- **Update Step:** Computes the Kalman Gain $\mathbf{K}_k$ and corrects the state estimate using the measurement residual $\mathbf{z}_k - \mathbf{H} \hat{\mathbf{x}}_k^-$.
- **Occlusion Handling:** If the target is physically occluded (no detection measurement $\mathbf{z}_k$), the predict step continues updating position based on estimated velocity $[v_x, v_y]$, maintaining smooth tracking through temporary dropouts.

---

## File Structure

```text
phase-02/07-object-tracking/
├── README.md
├── tracker.py     # Tracking algorithms (Kalman filter, CamShift, Lucas-Kanade)
└── main.py        # Real-time interactive tracking lab with split-screen HUD
```

---

## Running the Module

Execute via the project Makefile:

```bash
make run phase-02/07-object-tracking/main.py
```

Or directly via `uv`:

```bash
uv run python phase-02/07-object-tracking/main.py
```

### Interactive Controls
- **`1`:** Kalman Filter Tracker (smooth trajectory with occlusion prediction)
- **`2`:** Sparse Optical Flow (Lucas-Kanade motion tracking on Shi-Tomasi corners)
- **`3`:** Dense Optical Flow (Farnebäck motion vector field colorized via HSV)
- **`4`:** CamShift Tracker (adaptive color probability distribution tracking)
- **`Space`:** Reset tracking target or clear feature points
- **`c`:** Cycle camera device (0 <-> 2)
- **`s`:** Save snapshot and dump trajectory telemetry to `output/`
- **`q` or `Esc`:** Quit
