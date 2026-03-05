"""
Player detection using NVIDIA-accelerated models.

Strategy:
  1. Primary — NVIDIA PeopleNet (ONNX/TensorRT) via onnxruntime-gpu
  2. Fallback — YOLOv8 exported to TensorRT engine for NVIDIA GPU inference

Both paths leverage NVIDIA CUDA cores for real-time performance.
"""

import numpy as np
import cv2
from pathlib import Path
from config import (
    DETECTION_CONFIDENCE,
    NMS_IOU_THRESHOLD,
    USE_PEOPLENET,
    PEOPLENET_MODEL_PATH,
    YOLO_MODEL,
    YOLO_TRT_MODEL,
    MODELS_DIR,
)


class PeopleNetDetector:
    """NVIDIA PeopleNet detector via ONNX Runtime with CUDA Execution Provider."""

    def __init__(self, model_path: str = None):
        import onnxruntime as ort

        model_path = model_path or str(PEOPLENET_MODEL_PATH)
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)

        meta = self.session.get_inputs()[0]
        self.input_name = meta.name
        self.input_shape = meta.shape  # e.g. [1, 3, 544, 960]
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]

        print(f"[PeopleNet] Loaded model: {model_path}")
        print(f"[PeopleNet] Input shape: {self.input_shape}")
        print(f"[PeopleNet] Providers: {self.session.get_providers()}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return np.expand_dims(img, axis=0)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Returns list of {bbox: [x1,y1,x2,y2], confidence: float, class: 'person'}."""
        h, w = frame.shape[:2]
        blob = self.preprocess(frame)

        outputs = self.session.run(None, {self.input_name: blob})
        # PeopleNet outputs: [coverage, bboxes] — coverage is confidence grid,
        # bboxes are normalized coordinates per grid cell.
        coverage = outputs[0]  # [1, 3, grid_h, grid_w] — 3 classes: person, bag, face
        bboxes = outputs[1]    # [1, 12, grid_h, grid_w] — 4 coords * 3 classes

        detections = []
        person_coverage = coverage[0, 0]  # class 0 = person
        grid_h, grid_w = person_coverage.shape
        stride_y = self.input_h / grid_h
        stride_x = self.input_w / grid_w

        for gy in range(grid_h):
            for gx in range(grid_w):
                conf = person_coverage[gy, gx]
                if conf < DETECTION_CONFIDENCE:
                    continue

                # Bboxes for person class (indices 0-3)
                bx1 = (gx * stride_x - bboxes[0, 0, gy, gx]) * w / self.input_w
                by1 = (gy * stride_y - bboxes[0, 1, gy, gx]) * h / self.input_h
                bx2 = (gx * stride_x + bboxes[0, 2, gy, gx]) * w / self.input_w
                by2 = (gy * stride_y + bboxes[0, 3, gy, gx]) * h / self.input_h

                detections.append({
                    "bbox": [float(bx1), float(by1), float(bx2), float(by2)],
                    "confidence": float(conf),
                    "class": "person",
                })

        # Apply NMS
        if detections:
            detections = self._nms(detections)
        return detections

    def _nms(self, detections: list[dict]) -> list[dict]:
        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            DETECTION_CONFIDENCE, NMS_IOU_THRESHOLD,
        )
        if len(indices) == 0:
            return []
        return [detections[i] for i in indices.flatten()]


class YOLOTensorRTDetector:
    """YOLOv8 with NVIDIA TensorRT export for GPU-accelerated inference."""

    def __init__(self):
        from ultralytics import YOLO

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        trt_path = YOLO_TRT_MODEL
        if trt_path.exists():
            print(f"[YOLO-TRT] Loading TensorRT engine: {trt_path}")
            self.model = YOLO(str(trt_path))
        else:
            print(f"[YOLO-TRT] Loading PyTorch model: {YOLO_MODEL}")
            self.model = YOLO(YOLO_MODEL)
            # Export to TensorRT for NVIDIA GPU acceleration
            try:
                print("[YOLO-TRT] Exporting to TensorRT engine (first run only)...")
                export_path = self.model.export(format="engine", device=0)
                print(f"[YOLO-TRT] TensorRT engine saved: {export_path}")
                # Reload the TRT engine
                self.model = YOLO(export_path)
            except Exception as e:
                print(f"[YOLO-TRT] TensorRT export failed ({e}), using CUDA PyTorch fallback")
                self.model = YOLO(YOLO_MODEL)

        # YOLO class ID 0 = person in COCO
        self.person_class_id = 0

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model(frame, conf=DETECTION_CONFIDENCE, iou=NMS_IOU_THRESHOLD, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != self.person_class_id:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(box.conf[0]),
                    "class": "person",
                })

        return detections


def create_detector():
    """Factory: returns the best available NVIDIA-accelerated detector."""
    if USE_PEOPLENET and PEOPLENET_MODEL_PATH.exists():
        print("[Detector] Using NVIDIA PeopleNet")
        return PeopleNetDetector()

    print("[Detector] Using YOLOv8 with NVIDIA TensorRT acceleration")
    return YOLOTensorRTDetector()
