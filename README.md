# Soccer Player Tracker

[![Live Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-brightgreen)](https://garridolecca.github.io/Sport_Nvidia/)

Detect and track soccer players from match video using NVIDIA GPU-accelerated models,
then visualize their positions and movements on an interactive ArcGIS JavaScript API map.

**[Live Demo](https://garridolecca.github.io/Sport_Nvidia/)**

## Architecture

```
Match Video → NVIDIA Detection (PeopleNet/YOLOv8-TRT) → SORT Tracker → Homography → GPS coords → ArcGIS Map
```

## Quick Start (Demo Mode)

Run with simulated data to test the visualization immediately:

```bash
cd soccer-tracker/backend
pip install fastapi uvicorn

# Generate demo tracking data (no GPU needed)
python generate_demo_data.py

# Start the server
python server.py
```

Open http://localhost:8000 in your browser.

## Full Pipeline (NVIDIA GPU)

### 1. Install dependencies

```bash
cd soccer-tracker/backend
pip install -r requirements.txt
```

Requires:
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8+ installed
- cuDNN 8.x+

### 2. Download Alfheim dataset

```bash
python download_data.py
```

Or download manually from https://datasets.simula.no/alfheim/

### 3a. Process video with NVIDIA models

```bash
# Auto-detects and uses NVIDIA TensorRT acceleration
python pipeline.py data/2013-11-03/alfheim-2013-11-03_cam1.avi

# With interactive calibration (click pitch corners in GUI)
python pipeline.py data/2013-11-03/alfheim-2013-11-03_cam1.avi --interactive
```

### 3b. Or use ground-truth tracking data

```bash
python pipeline.py --zxy
```

### 4. Start the server

```bash
python server.py
```

Open http://localhost:8000

## NVIDIA Model Details

The pipeline supports two NVIDIA-accelerated detection backends:

| Model | Source | Speed | Accuracy |
|-------|--------|-------|----------|
| **PeopleNet** | NVIDIA NGC (ONNX) | ~30 FPS | High |
| **YOLOv8 + TensorRT** | Ultralytics → TRT export | ~60 FPS | High |

Both use NVIDIA CUDA/TensorRT for GPU inference. The TRT engine is built
on first run and cached for subsequent runs.

To use PeopleNet, download the ONNX model from NVIDIA NGC and place it at
`models/peoplenet.onnx`, then set `USE_PEOPLENET = True` in `config.py`.

## Frontend Features

- Real-time player position visualization on satellite imagery
- Play/pause/scrub timeline controls
- Adjustable playback speed (0.25x–8x)
- Player movement trails
- Density heatmap
- Click to select/highlight individual players
- Keyboard shortcuts: Space (play/pause), Arrow keys (step)

## Project Structure

```
soccer-tracker/
├── backend/
│   ├── config.py              # All configuration constants
│   ├── download_data.py       # Alfheim dataset downloader
│   ├── nvidia_detector.py     # NVIDIA PeopleNet + YOLOv8-TRT detection
│   ├── tracker.py             # SORT multi-object tracker
│   ├── homography.py          # Pixel → field → GPS coordinate mapping
│   ├── pipeline.py            # Full video processing pipeline
│   ├── generate_demo_data.py  # Simulated data for testing
│   ├── server.py              # FastAPI backend
│   └── requirements.txt
├── frontend/
│   ├── index.html             # ArcGIS JS API viewer
│   ├── app.js                 # Map visualization + playback
│   └── style.css
├── data/                      # Downloaded dataset files
├── output/                    # Processed tracking JSON
└── models/                    # NVIDIA model files
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/tracking` | Full tracking dataset |
| `GET /api/tracking/meta` | Field info + frame count |
| `GET /api/tracking/frames/{idx}` | Single frame data |
| `GET /api/tracking/range?start=0&end=100` | Frame range |
| `GET /api/tracking/player/{id}/trail` | Player movement trail |
| `GET /api/tracking/players` | All player IDs |

## Live Demo

The GitHub Pages demo runs entirely in the browser with embedded demo data (no backend needed):

**https://garridolecca.github.io/Sport_Nvidia/**

Features visible in the demo:
- Satellite basemap centered on Alfheim Stadium (Tromsoe, Norway)
- Soccer field overlay with all standard markings
- 22 players (Team A in red, Team B in blue) with animated movement
- Playback controls, trails, heatmap, and player selection
