"""
Simple IoU-based multi-object tracker (SORT-style).

Assigns persistent IDs to detected players across frames using
Hungarian algorithm matching on IoU scores.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from config import MAX_AGE, MIN_HITS, IOU_THRESHOLD


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of bounding boxes. Shape: (N,4) x (M,4) → (N,M)."""
    x1 = np.maximum(bb_test[:, 0:1], bb_gt[:, 0].T)
    y1 = np.maximum(bb_test[:, 1:2], bb_gt[:, 1].T)
    x2 = np.minimum(bb_test[:, 2:3], bb_gt[:, 2].T)
    y2 = np.minimum(bb_test[:, 3:4], bb_gt[:, 3].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_test = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_gt = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])

    union = area_test[:, None] + area_gt[None, :] - inter
    return inter / np.maximum(union, 1e-6)


class Track:
    _next_id = 1

    def __init__(self, bbox: np.ndarray):
        self.id = Track._next_id
        Track._next_id += 1
        self.bbox = bbox
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def update(self, bbox: np.ndarray):
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0

    def predict(self):
        """Simple constant-position prediction."""
        self.age += 1
        self.time_since_update += 1
        return self.bbox


class SORTTracker:
    """Simple Online Realtime Tracker."""

    def __init__(self):
        self.tracks: list[Track] = []

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Takes detections [{bbox, confidence, class}], returns tracked objects
        [{track_id, bbox, confidence}].
        """
        det_bboxes = np.array([d["bbox"] for d in detections]) if detections else np.empty((0, 4))

        # Predict existing tracks
        for t in self.tracks:
            t.predict()

        if len(self.tracks) == 0:
            # Initialize tracks from all detections
            for d in detections:
                self.tracks.append(Track(np.array(d["bbox"])))
        elif len(det_bboxes) > 0:
            track_bboxes = np.array([t.bbox for t in self.tracks])
            iou_matrix = iou_batch(det_bboxes, track_bboxes)

            # Hungarian matching (minimize cost = 1 - IoU)
            cost = 1.0 - iou_matrix
            row_idx, col_idx = linear_sum_assignment(cost)

            matched_det = set()
            matched_trk = set()

            for r, c in zip(row_idx, col_idx):
                if iou_matrix[r, c] >= IOU_THRESHOLD:
                    self.tracks[c].update(det_bboxes[r])
                    matched_det.add(r)
                    matched_trk.add(c)

            # Create new tracks for unmatched detections
            for i in range(len(det_bboxes)):
                if i not in matched_det:
                    self.tracks.append(Track(det_bboxes[i]))

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= MAX_AGE]

        # Return confirmed tracks
        results = []
        for t in self.tracks:
            if t.hits >= MIN_HITS or t.time_since_update == 0:
                results.append({
                    "track_id": t.id,
                    "bbox": t.bbox.tolist(),
                    "confidence": 1.0,
                })
        return results
