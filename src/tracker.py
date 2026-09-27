"""Trajectory and Track State Manager for Multi-Object Tracking."""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class TrackState:
    track_id: int
    class_name: str
    points: deque = field(default_factory=lambda: deque(maxlen=40))
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    total_detections: int = 0
    velocity_px_per_frame: float = 0.0


class TrajectoryTracker:
    def __init__(self, max_trajectory_length: int = 40):
        self.max_length = max_trajectory_length
        self.tracks: Dict[int, TrackState] = {}
        self.frame_index = 0

    def update(self, detections: list) -> None:
        """Update track histories from detections with assigned track_ids."""
        self.frame_index += 1
        current_active_ids = set()

        for det in detections:
            track_id = getattr(det, "track_id", None)
            if track_id is None:
                continue

            current_active_ids.add(track_id)
            x1, y1, x2, y2 = det.box
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            if track_id not in self.tracks:
                self.tracks[track_id] = TrackState(
                    track_id=track_id,
                    class_name=det.class_name,
                    points=deque(maxlen=self.max_length),
                    first_seen_frame=self.frame_index,
                    last_seen_frame=self.frame_index,
                    total_detections=1,
                )
            else:
                track = self.tracks[track_id]
                track.last_seen_frame = self.frame_index
                track.total_detections += 1

            track = self.tracks[track_id]
            if len(track.points) > 0:
                prev_x, prev_y = track.points[-1]
                dx = center[0] - prev_x
                dy = center[1] - prev_y
                track.velocity_px_per_frame = float(np.sqrt(dx * dx + dy * dy))

            track.points.append(center)

    def get_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """Return list of (x, y) coordinates for given track ID."""
        if track_id in self.tracks:
            return list(self.tracks[track_id].points)
        return []

    def get_active_count(self, max_idle_frames: int = 5) -> int:
        """Count currently active tracks within timeout."""
        return sum(
            1
            for t in self.tracks.values()
            if (self.frame_index - t.last_seen_frame) <= max_idle_frames
        )

    def get_summary(self) -> Dict[str, int]:
        """Aggregate total count per object class."""
        summary: Dict[str, int] = {}
        for t in self.tracks.values():
            summary[t.class_name] = summary.get(t.class_name, 0) + 1
        return summary

    def reset(self) -> None:
        """Clear all tracking state."""
        self.tracks.clear()
        self.frame_index = 0


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IOU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area <= 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = float(area1 + area2 - inter_area)

    return inter_area / union_area if union_area > 0 else 0.0


class SortTracker:
    """Classic Simple Online and Realtime Tracking (SORT) algorithm."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.frame_count = 0

    def update(self, detections: list) -> list:
        """Assign persistent track IDs to detections using IOU association."""
        self.frame_count += 1
        updated_detections = []
        unmatched_dets = list(range(len(detections)))
        matched_tracks = set()

        if self.tracks and detections:
            for track_id, track_info in list(self.tracks.items()):
                best_iou = self.iou_threshold
                best_det_idx = None
                for idx in unmatched_dets:
                    iou = compute_iou(track_info["box"], detections[idx].box)
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = idx

                if best_det_idx is not None:
                    matched_tracks.add(track_id)
                    unmatched_dets.remove(best_det_idx)
                    det = detections[best_det_idx]
                    det.track_id = track_id
                    track_info["box"] = det.box
                    track_info["last_frame"] = self.frame_count
                    updated_detections.append(det)

        for idx in unmatched_dets:
            det = detections[idx]
            det.track_id = self.next_id
            self.tracks[self.next_id] = {
                "box": det.box,
                "class_name": det.class_name,
                "last_frame": self.frame_count,
            }
            self.next_id += 1
            updated_detections.append(det)

        dead_ids = [
            tid
            for tid, tinfo in self.tracks.items()
            if (self.frame_count - tinfo["last_frame"]) > self.max_age
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        return updated_detections
