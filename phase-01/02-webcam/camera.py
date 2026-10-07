"""Thread-isolated camera ingestion pipeline.

Eliminates V4L2/OpenCV driver buffer bloat by running ingestion in a background
producer thread that continuously updates a single atomic frame slot.
"""

import os
import threading
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np


class FPSMeter:
    """Rolling FPS estimator using a bounded sliding window."""

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        """Record current frame tick and return estimated FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


class ThreadedCamera:
    """Threaded camera capture wrapper with zero-lag latest frame retrieval.

    Supports automatic fallback to synthetic animated test frames if no hardware
    webcam is detected.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        use_synthetic: bool = False,
    ) -> None:
        self.device_index = device_index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.use_synthetic = use_synthetic

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._frame_count: int = 0
        self._dropped_frames: int = 0

        self.cap: Optional[cv2.VideoCapture] = None
        self.backend_name = "Synthetic"

        if not self.use_synthetic:
            self._init_hardware_camera()

        if self.cap is None or not self.cap.isOpened():
            print("[Camera] No physical camera available. Falling back to Synthetic generator.")
            self.use_synthetic = True
            self.backend_name = "Synthetic (Animated)"

    def _init_hardware_camera(self) -> None:
        """Initialize OpenCV VideoCapture with Video4Linux backend on Linux."""
        backend = cv2.CAP_V4L2 if os.name == "posix" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.device_index, backend)

        if not self.cap.isOpened():
            return

        # Request MJPG codec for high resolution at 30+ FPS over USB
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        backend_id = int(self.cap.get(cv2.CAP_PROP_BACKEND))
        backend_tag = "V4L2" if backend == cv2.CAP_V4L2 else "OpenCV Default"
        self.backend_name = f"{backend_tag} (id={backend_id})"

    def start(self) -> "ThreadedCamera":
        """Start the background ingestion thread."""
        if self._running:
            return self

        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()

        # Wait briefly for first frame
        for _ in range(50):
            with self._lock:
                if self._frame is not None:
                    break
            time.sleep(0.01)

        return self

    def _generate_synthetic_frame(self, t: float) -> np.ndarray:
        """Generate an animated synthetic test frame with bouncing geometric target."""
        frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

        # Background subtle gradient
        ramp = np.linspace(20, 60, self.target_width, dtype=np.uint8)
        frame[:] = ramp[None, :, None]

        # Animated bouncing circle
        cx = int((np.sin(t * 2.0) * 0.4 + 0.5) * self.target_width)
        cy = int((np.cos(t * 2.5) * 0.3 + 0.5) * self.target_height)
        cv2.circle(frame, (cx, cy), 45, (0, 200, 255), -1)
        cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)

        # Crosshairs
        cv2.line(frame, (cx - 60, cy), (cx + 60, cy), (255, 255, 255), 1)
        cv2.line(frame, (cx, cy - 60), (cx, cy + 60), (255, 255, 255), 1)

        return frame

    def _capture_worker(self) -> None:
        """Background thread worker continuously draining camera frames."""
        target_period = 1.0 / max(1, self.target_fps)

        while self._running:
            start_time = time.perf_counter()

            if self.use_synthetic:
                frame = self._generate_synthetic_frame(start_time)
                ret = True
                sleep_dur = max(0.0, target_period - (time.perf_counter() - start_time))
                if sleep_dur > 0:
                    time.sleep(sleep_dur)
            else:
                ret, frame = self.cap.read() if self.cap else (False, None)

            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._timestamp = time.perf_counter()
                    self._frame_count += 1
            else:
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Retrieve the latest captured frame without queue latency.

        Returns:
            Tuple of (success, latest_frame_copy, capture_timestamp).
        """
        with self._lock:
            if self._frame is None:
                return False, None, 0.0
            # Return copy to ensure thread-safety against writer mutations
            return True, self._frame.copy(), self._timestamp

    def get_resolution(self) -> Tuple[int, int]:
        """Return actual (width, height) of current frame."""
        with self._lock:
            if self._frame is not None:
                h, w = self._frame.shape[:2]
                return w, h
        return self.target_width, self.target_height

    def release(self) -> None:
        """Stop background worker and release camera file descriptor."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __enter__(self) -> "ThreadedCamera":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
