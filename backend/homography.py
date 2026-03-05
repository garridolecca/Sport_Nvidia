"""
Homography: transform pixel coordinates → field coordinates → GPS coordinates.

The Alfheim dataset uses a fixed camera from midfield, making homography stable.
We define reference points on the pitch (corners, center circle, etc.) and compute
a perspective transform to map any pixel (x,y) to field meters, then to lat/lon.
"""

import numpy as np
import cv2
from config import (
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    STADIUM_LAT,
    STADIUM_LON,
    METERS_PER_DEG_LAT,
    METERS_PER_DEG_LON,
)


class FieldHomography:
    """Maps pixel coordinates to field coordinates and GPS."""

    def __init__(self):
        self.H = None  # 3x3 homography matrix

    def calibrate_from_points(
        self,
        pixel_points: np.ndarray,
        field_points: np.ndarray,
    ):
        """
        Compute homography from matched pixel ↔ field coordinate pairs.

        Args:
            pixel_points: (N, 2) array of pixel coordinates
            field_points: (N, 2) array of field coordinates in meters
                          where (0,0) = top-left corner of the pitch,
                          x = along length (0→105), y = along width (0→68)
        """
        assert len(pixel_points) >= 4, "Need at least 4 point correspondences"
        self.H, mask = cv2.findHomography(
            pixel_points.astype(np.float64),
            field_points.astype(np.float64),
            cv2.RANSAC, 5.0,
        )
        inliers = mask.sum() if mask is not None else 0
        print(f"[Homography] Computed with {inliers}/{len(pixel_points)} inliers")

    def calibrate_interactive(self, frame: np.ndarray):
        """
        Interactive calibration: user clicks on known pitch landmarks in the frame.
        Predefined field coordinates for standard landmarks are used.

        Standard landmarks (field coords in meters, origin = top-left corner):
          0: Top-left corner        (0, 0)
          1: Top-right corner       (105, 0)
          2: Bottom-right corner    (105, 68)
          3: Bottom-left corner     (0, 68)
          4: Center spot            (52.5, 34)
          5: Left penalty spot      (11, 34)
          6: Right penalty spot     (94, 34)
        """
        FIELD_LANDMARKS = np.array([
            [0, 0],
            [FIELD_LENGTH_M, 0],
            [FIELD_LENGTH_M, FIELD_WIDTH_M],
            [0, FIELD_WIDTH_M],
            [FIELD_LENGTH_M / 2, FIELD_WIDTH_M / 2],
            [11.0, FIELD_WIDTH_M / 2],
            [94.0, FIELD_WIDTH_M / 2],
        ], dtype=np.float64)

        LANDMARK_NAMES = [
            "Top-left corner", "Top-right corner",
            "Bottom-right corner", "Bottom-left corner",
            "Center spot", "Left penalty spot", "Right penalty spot",
        ]

        clicked = []

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked.append((x, y))
                idx = len(clicked) - 1
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                label = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"Point {idx}"
                cv2.putText(display, label, (x + 10, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Calibration", display)
                if len(clicked) >= 4:
                    print(f"  {len(clicked)} points marked. Press 'q' to finish or keep clicking.")

        display = frame.copy()
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Calibration", on_click)

        print("\n=== Interactive Calibration ===")
        print("Click on the following pitch landmarks in order:")
        for i, name in enumerate(LANDMARK_NAMES):
            print(f"  {i}: {name}")
        print("Press 'q' when done (minimum 4 points).\n")

        cv2.imshow("Calibration", display)
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q") and len(clicked) >= 4:
                break

        cv2.destroyAllWindows()

        n = min(len(clicked), len(FIELD_LANDMARKS))
        pixel_pts = np.array(clicked[:n], dtype=np.float64)
        field_pts = FIELD_LANDMARKS[:n]

        self.calibrate_from_points(pixel_pts, field_pts)

    def calibrate_default(self, frame_width: int, frame_height: int):
        """
        Default calibration assuming the camera shows roughly the full pitch.
        Good enough for prototyping with fixed wide-angle cameras.
        """
        pixel_points = np.array([
            [frame_width * 0.05, frame_height * 0.15],   # top-left
            [frame_width * 0.95, frame_height * 0.15],   # top-right
            [frame_width * 0.95, frame_height * 0.85],   # bottom-right
            [frame_width * 0.05, frame_height * 0.85],   # bottom-left
        ], dtype=np.float64)

        field_points = np.array([
            [0, 0],
            [FIELD_LENGTH_M, 0],
            [FIELD_LENGTH_M, FIELD_WIDTH_M],
            [0, FIELD_WIDTH_M],
        ], dtype=np.float64)

        self.calibrate_from_points(pixel_points, field_points)

    def pixel_to_field(self, px: float, py: float) -> tuple[float, float]:
        """Convert pixel (x, y) → field meters (fx, fy)."""
        if self.H is None:
            raise RuntimeError("Homography not calibrated. Call calibrate_* first.")
        pt = np.array([px, py, 1.0], dtype=np.float64)
        result = self.H @ pt
        result /= result[2]
        return float(result[0]), float(result[1])

    def field_to_gps(self, fx: float, fy: float) -> tuple[float, float]:
        """Convert field meters → GPS lat/lon using Alfheim Stadium anchor."""
        # fx runs along pitch length (East), fy runs along width (South)
        lon = STADIUM_LON + fx / METERS_PER_DEG_LON
        lat = STADIUM_LAT - fy / METERS_PER_DEG_LAT
        return lat, lon

    def pixel_to_gps(self, px: float, py: float) -> tuple[float, float]:
        """Pixel → GPS in one step."""
        fx, fy = self.pixel_to_field(px, py)
        return self.field_to_gps(fx, fy)

    def bbox_center_to_gps(self, bbox: list[float]) -> tuple[float, float]:
        """Get GPS of a bounding box's bottom-center (feet position)."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = bbox[3]  # bottom of bbox ≈ feet on ground
        return self.pixel_to_gps(cx, cy)
