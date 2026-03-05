"""FastAPI server to serve tracking data and static frontend files."""

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import OUTPUT_DIR, PROJECT_ROOT

app = FastAPI(title="Soccer Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache loaded tracking data
_tracking_cache = {}


def _load_tracking_data(filename: str = "tracking_data.json") -> dict:
    if filename in _tracking_cache:
        return _tracking_cache[filename]

    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, f"Tracking data not found: {filename}. Run pipeline.py first.")

    with open(filepath) as f:
        data = json.load(f)
    _tracking_cache[filename] = data
    return data


@app.get("/api/tracking")
def get_tracking_data(file: str = "tracking_data.json"):
    """Return full tracking dataset."""
    return _load_tracking_data(file)


@app.get("/api/tracking/meta")
def get_metadata(file: str = "tracking_data.json"):
    """Return field info and frame count (without frame data)."""
    data = _load_tracking_data(file)
    return {
        "source": data.get("source", data.get("video", "")),
        "total_frames": data["total_frames_processed"],
        "field": data["field"],
    }


@app.get("/api/tracking/frames/{frame_index}")
def get_frame(frame_index: int, file: str = "tracking_data.json"):
    """Return a single frame's tracking data by index."""
    data = _load_tracking_data(file)
    if frame_index < 0 or frame_index >= len(data["frames"]):
        raise HTTPException(404, f"Frame {frame_index} out of range")
    return data["frames"][frame_index]


@app.get("/api/tracking/range")
def get_frame_range(start: int = 0, end: int = 100, file: str = "tracking_data.json"):
    """Return a range of frames."""
    data = _load_tracking_data(file)
    frames = data["frames"][start:end]
    return {"field": data["field"], "frames": frames}


@app.get("/api/tracking/player/{player_id}/trail")
def get_player_trail(player_id: int, file: str = "tracking_data.json"):
    """Return all positions for a specific player (for drawing trails)."""
    data = _load_tracking_data(file)
    trail = []
    for frame in data["frames"]:
        for p in frame["players"]:
            if p["id"] == player_id:
                trail.append({
                    "timestamp": frame["timestamp"],
                    "lat": p["lat"],
                    "lon": p["lon"],
                    "field_x": p.get("field_x"),
                    "field_y": p.get("field_y"),
                })
                break
    if not trail:
        raise HTTPException(404, f"Player {player_id} not found")
    return trail


@app.get("/api/tracking/players")
def get_player_ids(file: str = "tracking_data.json"):
    """Return all unique player IDs."""
    data = _load_tracking_data(file)
    ids = set()
    for frame in data["frames"]:
        for p in frame["players"]:
            ids.add(p["id"])
    return sorted(ids)


# Serve frontend static files
frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)
