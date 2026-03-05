from pathlib import Path

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Alfheim Stadium, Tromsoe, Norway — real-world anchor coordinates
# The pitch is ~105m x 68m. We anchor the top-left corner of the pitch
# to these GPS coordinates and project outward.
FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0
STADIUM_LAT = 69.6496    # top-left corner latitude
STADIUM_LON = 18.9553    # top-left corner longitude

# Approximate metres-per-degree at Tromsoe latitude (~69.65 N)
# lat: 1 deg ~ 111,320 m (roughly constant)
# lon: 1 deg ~ 111,320 * cos(69.65) ~ 38,800 m
METERS_PER_DEG_LAT = 111_320.0
METERS_PER_DEG_LON = 38_800.0

# Video processing
TARGET_FPS = 10            # frames per second to process
DETECTION_CONFIDENCE = 0.4 # min confidence for player detection
NMS_IOU_THRESHOLD = 0.45   # non-max suppression IoU threshold

# NVIDIA model settings
# Primary: NVIDIA PeopleNet via TensorRT / ONNX Runtime
# Fallback: YOLOv8 with TensorRT export for NVIDIA GPU acceleration
USE_PEOPLENET = False       # set True if you have PeopleNet ONNX/TRT model
PEOPLENET_MODEL_PATH = MODELS_DIR / "peoplenet.onnx"
YOLO_MODEL = "yolov8n.pt"  # will auto-download; exports to TensorRT on first run
YOLO_TRT_MODEL = MODELS_DIR / "yolov8n.engine"

# Tracking
MAX_AGE = 30               # max frames to keep lost track
MIN_HITS = 3               # min detections before track is confirmed
IOU_THRESHOLD = 0.3        # IoU threshold for track association

# Server
API_HOST = "0.0.0.0"
API_PORT = 8000
