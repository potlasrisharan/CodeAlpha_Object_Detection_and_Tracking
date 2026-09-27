"""Object Detection Module using YOLOv8 with hardware acceleration."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


class ObjectDetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[str]] = None,
    ):
        self.device = self._resolve_device()
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or []
        self.class_names: Dict[int, str] = self.model.names

    @staticmethod
    def _resolve_device() -> str:
        """Select fastest available compute accelerator."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run single-frame inference without tracking."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

            if self.target_classes and cls_name not in self.target_classes:
                continue

            conf = float(box.conf[0].item())
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    confidence=round(conf, 3),
                    class_id=cls_id,
                    class_name=cls_name,
                )
            )

        return detections

    def track(self, frame: np.ndarray, persist: bool = True) -> List[Detection]:
        """Run inference with persistent multi-object tracking (ByteTrack / BoT-SORT)."""
        results = self.model.track(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            persist=persist,
            device=self.device,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        detections: List[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        has_ids = boxes.id is not None
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

            if self.target_classes and cls_name not in self.target_classes:
                continue

            conf = float(box.conf[0].item())
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

            track_id = int(boxes.id[i].item()) if has_ids else None

            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    confidence=round(conf, 3),
                    class_id=cls_id,
                    class_name=cls_name,
                    track_id=track_id,
                )
            )

        return detections
