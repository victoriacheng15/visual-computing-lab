"""Object tracking and temporal state estimation engine.

Provides implementations for:
1. 2D Linear Kalman Filter (constant-velocity kinematic state estimation).
2. Sparse Lucas-Kanade Optical Flow with Shi-Tomasi feature detection.
3. Dense Farneback Optical Flow motion field visualization.
4. Continuously Adaptive Mean-Shift (CamShift) color histogram tracker.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


class TrackingMode(Enum):
    """Supported tracking methods in the lab."""

    KALMAN = "Kalman Filter (Kinematic State)"
    LUCAS_KANADE = "Sparse Optical Flow (Lucas-Kanade)"
    FARNEBACK = "Dense Optical Flow (Farneback)"
    CAMSHIFT = "CamShift (Color Distribution)"


@dataclass
class TrackState:
    """Represents estimated kinematic state of an object."""

    measured_pos: Optional[Tuple[int, int]]
    estimated_pos: Tuple[int, int]
    velocity: Tuple[float, float]
    is_occluded: bool
    trajectory: List[Tuple[int, int]]


class LinearKalman2D:
    """Discrete linear 2D Kalman filter with constant-velocity motion model.

    State vector:   x = [x, y, vx, vy]^T
    Measurement:    z = [x, y]^T
    """

    def __init__(self, dt: float = 1.0 / 30.0) -> None:
        self.dt = dt
        # State dimension 4, measurement dimension 2
        self.kf = cv2.KalmanFilter(4, 2)

        # State transition matrix F
        self.kf.transitionMatrix = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Measurement matrix H
        self.kf.measurementMatrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Process noise covariance Q (kinematic acceleration uncertainty)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[2, 2] = 5.0
        self.kf.processNoiseCov[3, 3] = 5.0

        # Measurement noise covariance R (sensor detection noise)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        # Error covariance P
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.initialized = False
        self.history: Deque[Tuple[int, int]] = deque(maxlen=64)

    def predict(self) -> Tuple[int, int]:
        """Project state vector forward based on velocity."""
        pred = self.kf.predict()
        px = int(pred[0, 0])
        py = int(pred[1, 0])
        return px, py

    def update(self, measurement: Optional[Tuple[int, int]]) -> TrackState:
        """Execute Kalman predict and update cycle.

        Args:
            measurement: Raw (x, y) detection coordinate, or None if occluded.

        Returns:
            TrackState containing filtered position, velocity, and occlusion status.
        """
        # 1. Prediction step
        pred = self.kf.predict()
        px, py = int(pred[0, 0]), int(pred[1, 0])
        vx, vy = float(pred[2, 0]), float(pred[3, 0])

        is_occluded = measurement is None

        if measurement is not None:
            mx, my = measurement
            if not self.initialized:
                # Initialize state directly to first measurement
                self.kf.statePost = np.array([[mx], [my], [0.0], [0.0]], dtype=np.float32)
                self.initialized = True
                px, py = mx, my
            else:
                # 2. Correction step using measurement residual
                meas_mat = np.array([[np.float32(mx)], [np.float32(my)]])
                corrected = self.kf.correct(meas_mat)
                px, py = int(corrected[0, 0]), int(corrected[1, 0])
                vx, vy = float(corrected[2, 0]), float(corrected[3, 0])

        self.history.append((px, py))

        return TrackState(
            measured_pos=measurement,
            estimated_pos=(px, py),
            velocity=(vx, vy),
            is_occluded=is_occluded,
            trajectory=list(self.history),
        )

    def reset(self) -> None:
        """Reset internal filter state."""
        self.initialized = False
        self.history.clear()


class SparseOpticalFlowTracker:
    """Tracks point features using Lucas-Kanade with Shi-Tomasi corners."""

    def __init__(self, max_corners: int = 100) -> None:
        self.max_corners = max_corners
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7,
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.tracks: List[Deque[Tuple[int, int]]] = []

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sparse optical flow and render motion tracks.

        Returns:
            Tuple of (annotated_camera_frame, isolated_motion_trails_canvas).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()
        motion_canvas = np.full_like(frame, 20)  # Dark slate canvas

        if self.prev_gray is None or self.prev_points is None or len(self.prev_points) < 8:
            # Re-detect corners if points are sparse
            corners = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            if corners is not None:
                self.prev_points = corners
                self.tracks = [deque([(int(x), int(y))], maxlen=25) for [[x, y]] in corners]
            self.prev_gray = gray
            return vis, motion_canvas

        # Calculate optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )

        if next_points is not None and status is not None:
            good_new = next_points[status == 1]
            good_old = self.prev_points[status == 1]

            new_tracks: List[Deque[Tuple[int, int]]] = []
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = int(new[0]), int(new[1])
                c, d = int(old[0]), int(old[1])

                # Draw on camera feed
                cv2.line(vis, (a, b), (c, d), (0, 255, 0), 2)
                cv2.circle(vis, (a, b), 4, (0, 0, 255), -1)

                if i < len(self.tracks):
                    self.tracks[i].append((a, b))
                    new_tracks.append(self.tracks[i])

                    # Draw complete motion trail on motion_canvas
                    pts = list(self.tracks[i])
                    for t_idx in range(1, len(pts)):
                        cv2.line(motion_canvas, pts[t_idx - 1], pts[t_idx], (0, 255, 200), 2)
                    cv2.circle(motion_canvas, (a, b), 4, (0, 180, 255), -1)

            self.tracks = new_tracks
            self.prev_points = good_new.reshape(-1, 1, 2)
        else:
            self.prev_points = None

        self.prev_gray = gray
        return vis, motion_canvas

    def reset(self) -> None:
        """Clear cached feature points."""
        self.prev_gray = None
        self.prev_points = None
        self.tracks.clear()


class DenseOpticalFlowTracker:
    """Calculates full-frame Farneback motion field visualised in HSV."""

    def __init__(self) -> None:
        self.prev_gray: Optional[np.ndarray] = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Compute Farneback flow and convert motion vectors to BGR."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(frame)

        # Farneback flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Encode angle to Hue and magnitude to Value
        hsv = np.zeros_like(frame)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.prev_gray = gray
        return bgr_flow

    def reset(self) -> None:
        """Clear reference frame."""
        self.prev_gray = None


class CamShiftTracker:
    """Color histogram probability tracker with continuously adaptive search window."""

    def __init__(self) -> None:
        self.track_window: Optional[Tuple[int, int, int, int]] = None
        self.roi_hist: Optional[np.ndarray] = None
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def initialize(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """Extract HSV histogram from initial bounding box."""
        x, y, w, h = bbox
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Mask out low light / glare pixels
        mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        self.track_window = bbox

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Calculate backprojection and track search window."""
        vis = frame.copy()
        if self.roi_hist is None or self.track_window is None:
            return vis, None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # Apply CamShift
        ret, self.track_window = cv2.CamShift(dst, self.track_window, self.term_crit)

        # Draw rotated ellipse / box
        pts = cv2.boxPoints(ret).astype(np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        return vis, dst

    def reset(self) -> None:
        """Clear tracked target."""
        self.roi_hist = None
        self.track_window = None


def create_split_view(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    left_title: str = "Tracking Overlay",
    right_title: str = "Telemetry / Motion Map",
) -> np.ndarray:
    """Render side-by-side view for visual tracking feedback."""
    h, w = left_frame.shape[:2]
    half_w = w // 2
    resized_left = cv2.resize(left_frame, (half_w, h // 2))
    resized_right = cv2.resize(right_frame, (half_w, h // 2))

    cv2.putText(resized_left, left_title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(
        resized_right, right_title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
    )

    return np.hstack((resized_left, resized_right))
