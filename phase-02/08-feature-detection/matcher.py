"""Feature detection, descriptor matching, and planar homography engine.

Provides modular implementations for:
1. ORB (Oriented FAST and Rotated BRIEF) fast binary matching.
2. SIFT (Scale-Invariant Feature Transform) high-precision float matching.
3. Lowe's Ratio Test and RANSAC outlier rejection.
4. Planar homography estimation and Augmented Reality perspective warping.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np


class FeatureMode(Enum):
    """Supported feature matching and visualization modes."""

    ORB = "ORB (Binary Descriptors)"
    SIFT = "SIFT (Gradient Descriptors)"
    KEYPOINTS_ONLY = "Raw Keypoint Inspection"
    AR_WARP = "Augmented Reality Warp"


@dataclass
class HomographyResult:
    """Contains estimated homography and inlier geometry."""

    homography: Optional[np.ndarray]
    inlier_count: int
    total_matches: int
    corners_in_camera: Optional[np.ndarray]  # (4, 1, 2) transformed corners
    good_matches: List[cv2.DMatch]


class FeaturePipeline:
    """Extracts, matches, and estimates homography between reference and scene."""

    def __init__(self, max_features: int = 1000) -> None:
        self.max_features = max_features

        # ORB engine (binary)
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
        )
        self.bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # SIFT engine (floating-point)
        self.sift = cv2.SIFT_create(nfeatures=max_features)
        # FLANN matcher for fast float matching
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
        search_params = dict(checks=50)
        self.flann_sift = cv2.FlannBasedMatcher(index_params, search_params)

        # Reference target caching
        self.ref_image: Optional[np.ndarray] = None
        self.ref_gray: Optional[np.ndarray] = None
        self.ref_kps_orb: List[cv2.KeyPoint] = []
        self.ref_des_orb: Optional[np.ndarray] = None
        self.ref_kps_sift: List[cv2.KeyPoint] = []
        self.ref_des_sift: Optional[np.ndarray] = None

    def set_reference_image(self, bgr_image: np.ndarray) -> None:
        """Cache reference target image and pre-compute descriptors."""
        self.ref_image = bgr_image.copy()
        self.ref_gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # Precompute ORB
        kps_orb, des_orb = self.orb.detectAndCompute(self.ref_gray, None)
        self.ref_kps_orb = kps_orb
        self.ref_des_orb = des_orb

        # Precompute SIFT
        kps_sift, des_sift = self.sift.detectAndCompute(self.ref_gray, None)
        self.ref_kps_sift = kps_sift
        self.ref_des_sift = des_sift

    def find_homography(
        self,
        scene_gray: np.ndarray,
        use_sift: bool = False,
        ratio_threshold: float = 0.75,
        ransac_reproj_thresh: float = 3.0,
    ) -> Tuple[HomographyResult, List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Match reference to scene and solve for planar homography via RANSAC."""
        empty_result = HomographyResult(None, 0, 0, None, [])

        if self.ref_gray is None:
            return empty_result, [], None

        ref_des = self.ref_des_sift if use_sift else self.ref_des_orb
        ref_kps = self.ref_kps_sift if use_sift else self.ref_kps_orb

        if ref_des is None or len(ref_kps) < 4:
            return empty_result, [], None

        # Detect in current scene
        detector = self.sift if use_sift else self.orb
        scene_kps, scene_des = detector.detectAndCompute(scene_gray, None)

        if scene_des is None or len(scene_kps) < 4:
            return empty_result, scene_kps, scene_des

        # KNN matching (k=2 for Lowe's ratio test)
        try:
            matcher = self.flann_sift if use_sift else self.bf_orb
            knn_matches = matcher.knnMatch(ref_des, scene_des, k=2)
        except cv2.error:
            return empty_result, scene_kps, scene_des

        # Apply Lowe's Ratio Test
        good_matches: List[cv2.DMatch] = []
        for pair in knn_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            return (
                HomographyResult(None, 0, len(good_matches), None, good_matches),
                scene_kps,
                scene_des,
            )

        # Extract coordinates of matched pairs
        src_pts = np.float32([ref_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([scene_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Solve homography matrix with RANSAC
        h_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_thresh)

        if h_matrix is None or mask is None:
            return (
                HomographyResult(None, 0, len(good_matches), None, good_matches),
                scene_kps,
                scene_des,
            )

        inlier_count = int(np.sum(mask))

        # Project 4 reference corners into scene coordinates
        h_ref, w_ref = self.ref_gray.shape[:2]
        ref_corners = np.float32(
            [[0, 0], [w_ref - 1, 0], [w_ref - 1, h_ref - 1], [0, h_ref - 1]]
        ).reshape(-1, 1, 2)

        projected_corners = cv2.perspectiveTransform(ref_corners, h_matrix)

        inlier_matches = [good_matches[i] for i, m in enumerate(mask) if m[0] == 1]

        return (
            HomographyResult(
                homography=h_matrix,
                inlier_count=inlier_count,
                total_matches=len(good_matches),
                corners_in_camera=projected_corners,
                good_matches=inlier_matches,
            ),
            scene_kps,
            scene_des,
        )


def render_feature_visualization(
    frame: np.ndarray,
    pipeline: FeaturePipeline,
    mode: FeatureMode,
    h_result: HomographyResult,
    scene_kps: List[cv2.KeyPoint],
    ar_overlay_image: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render analytical overlay and right telemetry panel."""
    vis_left = frame.copy()

    # Create right panel with dark slate theme
    vis_right = np.full_like(frame, 22)
    h_f, w_f = frame.shape[:2]

    # Draw reference target thumbnail on top-right panel if exists
    if pipeline.ref_image is not None:
        thumb_w = 200
        thumb_h = int(pipeline.ref_image.shape[0] * (thumb_w / pipeline.ref_image.shape[1]))
        thumb_h = min(thumb_h, 150)
        thumb = cv2.resize(pipeline.ref_image, (thumb_w, thumb_h))
        vis_right[20 : 20 + thumb_h, 20 : 20 + thumb_w] = thumb
        cv2.rectangle(vis_right, (20, 20), (20 + thumb_w, 20 + thumb_h), (0, 255, 0), 1)
        cv2.putText(
            vis_right,
            "Reference Target",
            (20, thumb_h + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

    if mode == FeatureMode.KEYPOINTS_ONLY:
        # Draw rich keypoints with size and orientation angles on left frame
        vis_left = cv2.drawKeypoints(
            frame,
            scene_kps,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        # On right panel, keep target thumbnail and render keypoint telemetry
        cv2.putText(
            vis_right,
            f"Active Features: {len(scene_kps)} Keypoints",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 200),
            2,
        )
        cv2.putText(
            vis_right,
            "Circles indicate scale octave",
            (20, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )
        cv2.putText(
            vis_right,
            "Radial spokes indicate gradient orientation",
            (20, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )
        return vis_left, vis_right

    # If homography was found and corners exist
    if h_result.corners_in_camera is not None and h_result.inlier_count >= 10:
        corners_int = np.int32(h_result.corners_in_camera)

        if mode == FeatureMode.AR_WARP and ar_overlay_image is not None:
            # Warp virtual asset onto detected planar object
            h_ar, w_ar = ar_overlay_image.shape[:2]
            src_corners = np.float32(
                [[0, 0], [w_ar - 1, 0], [w_ar - 1, h_ar - 1], [0, h_ar - 1]]
            ).reshape(-1, 1, 2)

            h_warp, _ = cv2.findHomography(src_corners, h_result.corners_in_camera)
            if h_warp is not None:
                warped_ar = cv2.warpPerspective(ar_overlay_image, h_warp, (w_f, h_f))
                # Create mask of warped asset
                ar_mask = cv2.warpPerspective(
                    np.full((h_ar, w_ar), 255, dtype=np.uint8), h_warp, (w_f, h_f)
                )
                # Composite onto left frame
                vis_left[ar_mask > 0] = warped_ar[ar_mask > 0]
                # Draw bounding frame
                cv2.polylines(vis_left, [corners_int], True, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            # Draw green bounding quad around detected planar object
            cv2.polylines(vis_left, [corners_int], isClosed=True, color=(0, 255, 0), thickness=3)

            # Draw corner markers
            for pt in corners_int:
                cv2.circle(vis_left, tuple(pt[0]), 6, (0, 0, 255), -1)

    # Render telemetry stats on right panel
    text_y = 220
    status_str = "LOCKED" if (h_result.inlier_count >= 10) else "SEARCHING"
    status_color = (0, 255, 0) if (h_result.inlier_count >= 10) else (0, 100, 255)

    cv2.putText(
        vis_right,
        f"Tracking State : {status_str}",
        (20, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
    )
    cv2.putText(
        vis_right,
        f"Inliers / RANSAC: {h_result.inlier_count:4d} / {h_result.total_matches:4d}",
        (20, text_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        vis_right,
        f"Scene Keypoints : {len(scene_kps):4d}",
        (20, text_y + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        vis_right,
        "Controls:",
        (20, text_y + 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 255, 120),
        1,
    )
    cv2.putText(
        vis_right,
        "- Press SPACE to capture new target",
        (20, text_y + 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        vis_right,
        "- Keys 1-4 switch algorithms/AR",
        (20, text_y + 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )

    return vis_left, vis_right


def create_split_view(
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    left_title: str = "Planar Recognition Feed",
    right_title: str = "Feature Telemetry",
) -> np.ndarray:
    """Render side-by-side view for interactive feedback."""
    h, w = left_frame.shape[:2]
    half_w = w // 2
    resized_left = cv2.resize(left_frame, (half_w, h // 2))
    resized_right = cv2.resize(right_frame, (half_w, h // 2))

    cv2.putText(resized_left, left_title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(
        resized_right, right_title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
    )

    return np.hstack((resized_left, resized_right))
