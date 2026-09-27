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
