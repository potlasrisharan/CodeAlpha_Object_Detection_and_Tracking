"""Streamlit Telemetry Dashboard for Real-Time Object Detection & Tracking."""

import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.detector import ObjectDetector
from src.tracker import TrajectoryTracker
from src.visualizer import FrameVisualizer
from src.utils import FPSCalculator, generate_synthetic_demo_video

st.set_page_config(
    page_title="Telemetry Vision / Object Detection & Tracking",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Taste CSS complying with stitch-design-taste and design-taste-frontend
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #09090b !important;
        color: #f4f4f5 !important;
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Telemetry Header */
    .telemetry-header {
        border-bottom: 1px solid #27272a;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }
    .telemetry-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: #f4f4f5;
        margin: 0;
    }
    .telemetry-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #10b981;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
    }

    /* Metric Bento */
    .metric-card {
        background-color: #121215;
        border: 1px solid #27272a;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        transition: border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #3f3f46;
    }
    .metric-label {
        font-family: 'Geist', sans-serif;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #a1a1aa;
        margin-bottom: 0.25rem;
    }
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #f4f4f5;
        letter-spacing: -0.03em;
    }

    /* Control Panel */
    .stSidebar {
        background-color: #0d0d10 !important;
        border-right: 1px solid #27272a;
    }
    .stSidebar [data-testid="stMarkdownContainer"] p {
        font-size: 0.85rem;
        color: #a1a1aa;
    }

    /* Buttons */
    .stButton > button {
        background-color: #10b981 !important;
        color: #09090b !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        border-radius: 0.5rem !important;
        border: none !important;
        padding: 0.55rem 1rem !important;
        transition: transform 0.1s ease, background-color 0.15s ease !important;
    }
    .stButton > button:hover {
        background-color: #059669 !important;
    }
    .stButton > button:active {
        transform: scale(0.98) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="telemetry-header">
        <div>
            <h1 class="telemetry-title">Real-Time Object Detection & Tracking</h1>
            <p style="color: #71717a; font-size: 0.85rem; margin-top: 0.25rem;">
                Autonomous multi-object tracking architecture using YOLOv8 & ByteTrack.
            </p>
        </div>
        <div>
            <span class="telemetry-tag">ENGINE ACTIVE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
st.sidebar.markdown("### Detection Controls")

model_choice = st.sidebar.selectbox(
    "Model Checkpoint",
    options=["yolov8n.pt", "yolov8s.pt"],
    index=0,
    help="Select nano or small model based on throughput requirements",
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.40,
    step=0.05,
)

iou_threshold = st.sidebar.slider(
    "IOU Overlap Threshold",
    min_value=0.20,
    max_value=0.80,
    value=0.45,
    step=0.05,
)

show_trajectories = st.sidebar.checkbox("Render Trajectory Trails", value=True)
show_hud_overlay = st.sidebar.checkbox("Overlay HUD Stats on Video", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Video Stream Input")
source_mode = st.sidebar.radio(
    "Select Input Source",
    options=["Synthetic Traffic Demo", "Upload Video File", "Live Camera"],
    index=0,
)

# Initialize Detector and Tracker
@st.cache_resource
def load_detector(model_name: str, conf: float, iou: float):
    return ObjectDetector(model_name=model_name, confidence_threshold=conf, iou_threshold=iou)

detector = load_detector(model_choice, conf_threshold, iou_threshold)
# Update thresholds dynamically
detector.confidence_threshold = conf_threshold
detector.iou_threshold = iou_threshold

tracker = TrajectoryTracker(max_trajectory_length=40)
visualizer = FrameVisualizer(show_trajectories=show_trajectories, show_hud=show_hud_overlay)
fps_calc = FPSCalculator()

# Input source handling
video_cap = None
temp_file_path = None

if source_mode == "Synthetic Traffic Demo":
    demo_path = generate_synthetic_demo_video("sample_traffic.mp4")
    video_cap = cv2.VideoCapture(demo_path)
elif source_mode == "Upload Video File":
    uploaded = st.sidebar.file_uploader("Upload video file (mp4, mov, avi)", type=["mp4", "mov", "avi"])
    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        temp_file_path = tfile.name
        video_cap = cv2.VideoCapture(temp_file_path)
    else:
        st.info("Upload a video file from the sidebar to start inference.")
elif source_mode == "Live Camera":
    video_cap = cv2.VideoCapture(0)

# Layout: Split View (Main Stream on left / Telemetry metrics & log on right)
col_stream, col_metrics = st.columns([1.5, 1], gap="medium")

with col_stream:
    video_placeholder = st.empty()
    run_btn = st.button("Start Telemetry Stream")

with col_metrics:
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        fps_metric = st.empty()
    with m_col2:
        active_metric = st.empty()

    m_col3, m_col4 = st.columns(2)
    with m_col3:
        unique_metric = st.empty()
    with m_col4:
        device_metric = st.empty()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    table_placeholder = st.empty()

# Static metrics display
device_metric.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Compute Device</div>
        <div class="metric-val" style="color: #3b82f6;">{detector.device.upper()}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if run_btn and video_cap is not None:
    tracker.reset()
    stop_btn = st.sidebar.button("Stop Inference Stream")

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            if source_mode == "Synthetic Traffic Demo":
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        fps = fps_calc.tick()
        detections = detector.track(frame, persist=True)
        tracker.update(detections)

        annotated = visualizer.draw(frame, detections, tracker, fps)
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        video_placeholder.image(frame_rgb, use_container_width=True)

        # Update Live Telemetry Metrics
        fps_metric.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Inference FPS</div>
                <div class="metric-val" style="color: #10b981;">{fps:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        active_count = tracker.get_active_count()
        active_metric.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Active Tracks</div>
                <div class="metric-val">{active_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        unique_count = len(tracker.tracks)
        unique_metric.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Unique Objects</div>
                <div class="metric-val">{unique_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Update detection log
        if detections:
            data = [
                {
                    "Track ID": f"#{d.track_id}" if d.track_id is not None else "Unassigned",
                    "Class": d.class_name,
                    "Confidence": f"{d.confidence * 100:.1f}%",
                    "Coordinates": f"({d.box[0]}, {d.box[1]}) -> ({d.box[2]}, {d.box[3]})",
                }
                for d in detections[:8]
            ]
            df = pd.DataFrame(data)
            table_placeholder.dataframe(df, use_container_width=True, hide_index=True)

        time.sleep(0.01)

    video_cap.release()
