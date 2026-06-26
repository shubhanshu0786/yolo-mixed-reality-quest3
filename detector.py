import time
import numpy as np
import cv2
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path="yolo11n.pt", confidence=0.5, iou_threshold=0.45,
                 device="cpu", filter_classes=None, max_detections=100):
        print(f"Loading model: {model_path} on device: {device}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.filter_classes = filter_classes
        self.max_detections = max_detections
        # Warm up the model so first real frame isn't slow
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print("Model ready.")

    def detect(self, frame: np.ndarray) -> dict:
        t_start = time.perf_counter()

        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            classes=self.filter_classes,
            max_det=self.max_detections,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                    },
                    "center": {
                        "x": round((x1 + x2) / 2, 1),
                        "y": round((y1 + y2) / 2, 1),
                    },
                    "size": {
                        "width": round(x2 - x1, 1),
                        "height": round(y2 - y1, 1),
                    },
                })

        return {
            "detections": detections,
            "inference_time_ms": round(elapsed_ms, 2),
            "frame_size": {"width": frame.shape[1], "height": frame.shape[0]},
        }

    def annotate(self, frame: np.ndarray, result: dict) -> np.ndarray:
        annotated = frame.copy()
        for det in result["detections"]:
            b = det["bbox"]
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        return annotated
