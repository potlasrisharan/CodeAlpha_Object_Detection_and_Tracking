"""CLI Runner for Real-Time Object Detection and Tracking."""

import argparse
import sys
from pathlib import Path
import cv2
from src.detector import ObjectDetector
from src.tracker import TrajectoryTracker, SortTracker
from src.visualizer import FrameVisualizer
from src.utils import FPSCalculator, generate_synthetic_demo_video


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Object Detection & Tracking with YOLOv8 & Deep SORT / ByteTrack / SORT")
    parser.add_argument(
        "--source",
        type=str,
        default="sample_feed.mp4",
        help="Input source: '0' for webcam, path to video file, or 'demo' for synthetic traffic simulation",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort",
        choices=["botsort", "bytetrack", "sort"],
        help="Tracking algorithm: 'botsort' (Deep SORT), 'bytetrack', or 'sort'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path or model identifier (e.g., yolov8n.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence detection threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IOU threshold",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=[],
        help="Filter classes (e.g. --classes person car bicycle)",
    )
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Disable drawing trajectory movement lines",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default="",
        help="Path to save annotated output video (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine video source
    if args.source.lower() == "demo":
        print("[INFO] Generating synthetic test video feed...")
        source_path = generate_synthetic_demo_video("sample_traffic.mp4")
        cap = cv2.VideoCapture(source_path)
    elif args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        if not Path(args.source).exists():
            print(f"[ERROR] Source file not found: {args.source}", file=sys.stderr)
            sys.exit(1)
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source: {args.source}", file=sys.stderr)
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Optional video writer
    writer = None
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_output, fourcc, fps_in, (width, height))
        print(f"[INFO] Saving processed video to: {args.save_output}")

    print("[INFO] Initializing YOLOv8 Object Detector...")
    detector = ObjectDetector(
        model_name=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        target_classes=args.classes,
    )
    tracker = TrajectoryTracker(max_trajectory_length=40)
    visualizer = FrameVisualizer(show_trajectories=not args.no_trajectories, show_hud=True)
    fps_calc = FPSCalculator()
    sort_engine = SortTracker(iou_threshold=args.iou)
    tracker_yaml = f"{args.tracker}.yaml" if args.tracker in ["botsort", "bytetrack"] else "botsort.yaml"

    print(f"[INFO] Running on compute accelerator: {detector.device.upper()}")
    print(f"[INFO] Active tracking algorithm: {args.tracker.upper()}")
    print("[INFO] Press 'q' inside video window to exit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # If demo video, loop back
                if args.source.lower() == "demo":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            fps = fps_calc.tick()
            if args.tracker == "sort":
                raw_dets = detector.detect(frame)
                detections = sort_engine.update(raw_dets)
            else:
                detections = detector.track(frame, persist=True, tracker_algorithm=tracker_yaml)

            tracker.update(detections)

            annotated_frame = visualizer.draw(frame, detections, tracker, fps)

            if writer is not None:
                writer.write(annotated_frame)

            cv2.imshow("Real-Time Object Detection and Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Session summary:")
        summary = tracker.get_summary()
        for cls_name, count in summary.items():
            print(f"  - {cls_name}: {count} unique tracks")


if __name__ == "__main__":
    main()
