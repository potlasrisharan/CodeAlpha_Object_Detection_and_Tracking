"""Sleek OpenCV Telemetry Visualizer following anti-slop design system."""

from typing import List, Tuple, Dict
import cv2
import numpy as np
from src.detector import Detection
from src.tracker import TrajectoryTracker

# Calibrated BGR palette (Zinc + Emerald + Signal Blue + Amber)
PALETTE = {
    "emerald": (129, 185, 16),      # #10B981
    "signal_blue": (246, 130, 59),   # #3B82F6
    "amber": (11, 158, 245),         # #F59E0B
    "rose": (72, 29, 225),           # #E11D48
    "hud_bg": (21, 18, 18),          # #121215
    "hud_border": (42, 39, 39),      # #27272A
    "text_primary": (245, 244, 244), # #F4F4F5
    "text_muted": (170, 161, 161),   # #A1A1AA
}

CLASS_COLOR_MAP = {
    "person": PALETTE["emerald"],
    "car": PALETTE["signal_blue"],
    "truck": PALETTE["signal_blue"],
    "bus": PALETTE["signal_blue"],
    "motorcycle": PALETTE["amber"],
    "bicycle": PALETTE["amber"],
}


class FrameVisualizer:
    def __init__(self, show_trajectories: bool = True, show_hud: bool = True):
        self.show_trajectories = show_trajectories
        self.show_hud = show_hud

    def _get_color(self, class_name: str, track_id: int = None) -> Tuple[int, int, int]:
        if class_name in CLASS_COLOR_MAP:
            return CLASS_COLOR_MAP[class_name]
        if track_id is not None:
            colors = [PALETTE["emerald"], PALETTE["signal_blue"], PALETTE["amber"], PALETTE["rose"]]
            return colors[track_id % len(colors)]
        return PALETTE["emerald"]

    def draw(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        tracker: TrajectoryTracker,
        fps: float,
    ) -> np.ndarray:
        """Annotate frame with bounding boxes, badges, trajectories, and HUD."""
        annotated = frame.copy()

        # 1. Draw trajectory paths
        if self.show_trajectories:
            for det in detections:
                if det.track_id is None:
                    continue
                points = tracker.get_trajectory(det.track_id)
                if len(points) < 2:
                    continue

                color = self._get_color(det.class_name, det.track_id)
                for i in range(1, len(points)):
                    alpha = float(i) / len(points)
                    thickness = max(1, int(round(alpha * 3)))
                    cv2.line(annotated, points[i - 1], points[i], color, thickness, cv2.LINE_AA)

        # 2. Draw detections & pill badges
        for det in detections:
            x1, y1, x2, y2 = det.box
            color = self._get_color(det.class_name, det.track_id)

            # Clean bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Subdued corner accents
            corner_len = min(16, (x2 - x1) // 4, (y2 - y1) // 4)
            if corner_len > 4:
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, 3, cv2.LINE_AA)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, 3, cv2.LINE_AA)
                cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, 3, cv2.LINE_AA)
                cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, 3, cv2.LINE_AA)
                cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, 3, cv2.LINE_AA)
                cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, 3, cv2.LINE_AA)
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, 3, cv2.LINE_AA)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, 3, cv2.LINE_AA)

            # Pill badge label
            id_str = f"#{det.track_id} " if det.track_id is not None else ""
            label_text = f"{id_str}{det.class_name} {int(det.confidence * 100)}%"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            badge_y1 = max(0, y1 - text_h - 10)
            badge_y2 = badge_y1 + text_h + 8
            badge_x1 = x1
            badge_x2 = x1 + text_w + 14

            # Badge background
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), PALETTE["hud_bg"], -1)
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1, cv2.LINE_AA)

            # Badge text
            cv2.putText(
                annotated,
                label_text,
                (badge_x1 + 7, badge_y2 - 5),
                font,
                font_scale,
                PALETTE["text_primary"],
                thickness,
                cv2.LINE_AA,
            )

        # 3. Telemetry HUD Bar
        if self.show_hud:
            h, w = annotated.shape[:2]
            hud_h = 38
            hud_overlay = annotated.copy()
            cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), PALETTE["hud_bg"], -1)
            cv2.line(hud_overlay, (0, hud_h), (w, hud_h), PALETTE["hud_border"], 1)
            cv2.addWeighted(hud_overlay, 0.92, annotated, 0.08, 0, annotated)

            active_tracks = tracker.get_active_count()
            total_unique = len(tracker.tracks)
            hud_items = [
                f"FPS: {fps:.1f}",
                f"DETECTED: {len(detections)}",
                f"ACTIVE TRACKS: {active_tracks}",
                f"TOTAL UNIQUE: {total_unique}",
            ]

            x_offset = 18
            for item in hud_items:
                cv2.putText(
                    annotated,
                    item,
                    (x_offset, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    PALETTE["text_primary"],
                    1,
                    cv2.LINE_AA,
                )
                x_offset += 190

        return annotated
