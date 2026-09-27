"""Utility helpers for FPS calculation, video processing, and synthetic sample generation."""

import time
from collections import deque
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np


class FPSCalculator:
    def __init__(self, buffer_size: int = 30):
        self.timestamps = deque(maxlen=buffer_size)

    def tick(self) -> float:
        now = time.perf_counter()
        self.timestamps.append(now)
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return float((len(self.timestamps) - 1) / elapsed)


def generate_synthetic_demo_video(
    output_path: str = "sample_traffic.mp4",
    num_frames: int = 180,
    fps: int = 30,
    resolution: Tuple[int, int] = (960, 540),
) -> str:
    """Generate a clean synthetic video simulation with moving vehicles and pedestrians."""
    width, height = resolution
    path = Path(output_path)
    if path.exists() and path.stat().st_size > 1000:
        return str(path.resolve())

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    # Static road and background
    base_canvas = np.full((height, width, 3), (24, 24, 27), dtype=np.uint8)  # Zinc-900
    # Road lanes
    road_top = height // 3
    road_bot = height * 4 // 5
    cv2.rectangle(base_canvas, (0, road_top), (width, road_bot), (39, 39, 42), -1)  # Zinc-800
    cv2.line(base_canvas, (0, road_top), (width, road_top), (82, 82, 91), 2)
    cv2.line(base_canvas, (0, road_bot), (width, road_bot), (82, 82, 91), 2)

    # Dashed center line
    dash_w = 40
    lane_y = (road_top + road_bot) // 2
    for x in range(0, width, dash_w * 2):
        cv2.line(base_canvas, (x, lane_y), (x + dash_w, lane_y), (113, 113, 122), 2)

    # Object movement states
    car1_x = -120
    car2_x = width + 50
    ped_x = 200
    ped_y = 60

    for _ in range(num_frames):
        frame = base_canvas.copy()

        # Update Car 1 (moving right)
        car1_x += 6
        if car1_x > width + 100:
            car1_x = -150
        c1_w, c1_h = 130, 55
        c1_y = road_top + 25
        cv2.rectangle(frame, (car1_x, c1_y), (car1_x + c1_w, c1_y + c1_h), (59, 130, 246), -1)
        cv2.rectangle(frame, (car1_x + 30, c1_y + 10), (car1_x + 90, c1_y + 40), (24, 24, 27), -1)

        # Update Car 2 (moving left)
        car2_x -= 5
        if car2_x < -150:
            car2_x = width + 100
        c2_w, c2_h = 140, 60
        c2_y = lane_y + 20
        cv2.rectangle(frame, (car2_x, c2_y), (car2_x + c2_w, c2_y + c2_h), (245, 158, 11), -1)

        # Update Pedestrian (moving diagonally across crosswalk)
        ped_x += 1
        ped_y += 2
        if ped_y > road_bot + 40:
            ped_y = 50
            ped_x = 180
        cv2.circle(frame, (ped_x, ped_y), 14, (16, 185, 129), -1)
        cv2.rectangle(frame, (ped_x - 10, ped_y + 14), (ped_x + 10, ped_y + 46), (16, 185, 129), -1)

        out.write(frame)

    out.release()
    return str(path.resolve())
