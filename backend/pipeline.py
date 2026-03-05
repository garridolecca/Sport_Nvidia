"""
Main processing pipeline:
  Video → Frame Extraction → NVIDIA Detection → Tracking → Homography → JSON output

Can also load Alfheim ZXY ground-truth data directly for comparison/fallback.
"""

import json
import csv
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import DATA_DIR, OUTPUT_DIR, TARGET_FPS
from nvidia_detector import create_detector
from tracker import SORTTracker
from homography import FieldHomography


def process_video(
    video_path: str,
    output_path: str = None,
    calibrate_interactive: bool = False,
    max_frames: int = None,
):
    """
    Full pipeline: process a match video and output tracked player GPS positions.

    Args:
        video_path: path to input video file
        output_path: path for output JSON (default: output/tracking_data.json)
        calibrate_interactive: if True, opens GUI for manual pitch calibration
        max_frames: limit number of frames to process (None = all)
    """
    output_path = output_path or str(OUTPUT_DIR / "tracking_data.json")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_skip = max(1, int(video_fps / TARGET_FPS))

    print(f"Video: {video_path}")
    print(f"  Resolution: {frame_w}x{frame_h}, FPS: {video_fps:.1f}")
    print(f"  Total frames: {total_frames}, processing every {frame_skip}th frame")

    # Initialize NVIDIA detector
    detector = create_detector()

    # Initialize tracker
    tracker = SORTTracker()

    # Initialize homography
    homography = FieldHomography()
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")

    if calibrate_interactive:
        homography.calibrate_interactive(first_frame)
    else:
        homography.calibrate_default(frame_w, frame_h)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Process frames
    all_frames = []
    frame_idx = 0
    processed = 0

    pbar = tqdm(total=min(total_frames, max_frames or total_frames), desc="Processing")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        if frame_idx % frame_skip == 0:
            timestamp = frame_idx / video_fps

            # Detect players using NVIDIA model
            detections = detector.detect(frame)

            # Track across frames
            tracked = tracker.update(detections)

            # Convert to GPS coordinates
            players = []
            for t in tracked:
                lat, lon = homography.bbox_center_to_gps(t["bbox"])
                fx, fy = homography.pixel_to_field(
                    (t["bbox"][0] + t["bbox"][2]) / 2,
                    t["bbox"][3],
                )
                players.append({
                    "id": t["track_id"],
                    "lat": round(lat, 7),
                    "lon": round(lon, 7),
                    "field_x": round(fx, 2),
                    "field_y": round(fy, 2),
                    "bbox": [round(v, 1) for v in t["bbox"]],
                    "confidence": round(t["confidence"], 3),
                })

            all_frames.append({
                "frame": frame_idx,
                "timestamp": round(timestamp, 3),
                "players": players,
            })
            processed += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # Save output
    output_data = {
        "video": str(video_path),
        "video_fps": video_fps,
        "target_fps": TARGET_FPS,
        "total_frames_processed": processed,
        "field": {
            "length_m": 105,
            "width_m": 68,
            "anchor_lat": homography.field_to_gps(0, 0)[0],
            "anchor_lon": homography.field_to_gps(0, 0)[1],
            "corner_tl": list(homography.field_to_gps(0, 0)),
            "corner_tr": list(homography.field_to_gps(105, 0)),
            "corner_br": list(homography.field_to_gps(105, 68)),
            "corner_bl": list(homography.field_to_gps(0, 68)),
        },
        "frames": all_frames,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {processed} frames with tracking data to: {output_path}")
    return output_data


def load_alfheim_zxy(csv_path: str = None, output_path: str = None):
    """
    Load Alfheim ZXY ground-truth tracking data and convert to our JSON format.

    The ZXY system provides real positions at 20Hz. Format:
      timestamp, tag_id, x_pos, y_pos, heading, speed, ...

    This serves as ground truth and also as a fallback if video processing
    is not available (e.g., no NVIDIA GPU).
    """
    csv_path = csv_path or str(DATA_DIR / "2013-11-03" / "alfheim-2013-11-03_zxy.csv")
    output_path = output_path or str(OUTPUT_DIR / "tracking_data.json")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Alfheim ZXY data from: {csv_path}")

    homography = FieldHomography()

    frames_dict = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)

        for row in reader:
            if len(row) < 4:
                continue
            try:
                timestamp = float(row[0])
                tag_id = int(row[1])
                x_pos = float(row[2])  # field x in meters
                y_pos = float(row[3])  # field y in meters
            except (ValueError, IndexError):
                continue

            # Quantize to ~10 FPS (every 0.1s)
            frame_key = round(timestamp * 10)

            if frame_key not in frames_dict:
                frames_dict[frame_key] = {
                    "frame": frame_key,
                    "timestamp": round(timestamp, 3),
                    "players": [],
                }

            lat, lon = homography.field_to_gps(x_pos, y_pos)
            frames_dict[frame_key]["players"].append({
                "id": tag_id,
                "lat": round(lat, 7),
                "lon": round(lon, 7),
                "field_x": round(x_pos, 2),
                "field_y": round(y_pos, 2),
                "confidence": 1.0,
            })

    frames = sorted(frames_dict.values(), key=lambda f: f["frame"])

    output_data = {
        "source": "alfheim_zxy_ground_truth",
        "total_frames_processed": len(frames),
        "field": {
            "length_m": 105,
            "width_m": 68,
            "anchor_lat": homography.field_to_gps(0, 0)[0],
            "anchor_lon": homography.field_to_gps(0, 0)[1],
            "corner_tl": list(homography.field_to_gps(0, 0)),
            "corner_tr": list(homography.field_to_gps(105, 0)),
            "corner_br": list(homography.field_to_gps(105, 68)),
            "corner_bl": list(homography.field_to_gps(0, 68)),
        },
        "frames": frames,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(frames)} frames to: {output_path}")
    return output_data


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--zxy":
        # Load ground truth directly
        load_alfheim_zxy()
    else:
        # Process video with NVIDIA models
        video = str(DATA_DIR / "2013-11-03" / "alfheim-2013-11-03_cam1.avi")
        if len(sys.argv) > 1:
            video = sys.argv[1]
        process_video(video, calibrate_interactive="--interactive" in sys.argv)
