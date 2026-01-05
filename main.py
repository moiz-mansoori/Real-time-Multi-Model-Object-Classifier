"""
Real-time Multi-Model Object Classifier
========================================
Main entry point for the Lab-14 Complex Computing Activity project.

This project integrates concepts from:
- Lab 1: Data Preprocessing (image resizing, normalization)
- Lab 5: Ensemble/Multiple Models (YOLOv8 + YOLOv5)
- Lab 6: Model Comparison & Optimization (FPS, latency comparison)
- Lab 12: Real-Time ML Pipeline (live camera feed)

Course: Machine Learning (BSAI-462)
Lab: 14 - Complex Computing Activity

Usage:
    python main.py --model yolo8 --source 0
    python main.py --model yolo5 --source "http://192.168.1.x:8080/video"
    python main.py --model both --source 0

Controls:
    1 - Switch to YOLOv8
    2 - Switch to YOLOv5
    + - Increase confidence threshold
    - - Decrease confidence threshold
    Q - Quit
"""

import argparse
import sys
import cv2
import numpy as np
import time

# Add src to path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.camera.video_stream import VideoStream, create_ip_webcam_url
from src.preprocessing.image_processor import ImageProcessor
from src.detection.yolo_v8_detector import YOLOv8Detector
from src.detection.yolo_v5_detector import YOLOv5Detector
from src.utils.config import Config
from src.utils.performance_metrics import PerformanceMetrics
from src.utils.visualization import Visualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time Multi-Model Object Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model yolo8 --source 0
  python main.py --model yolo5 --source "http://192.168.1.10:8080/video"
  python main.py --model both --source test_video.mp4
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolo8",
        choices=["yolo8", "yolo5", "both"],
        help="Model to use: yolo8, yolo5, or both for comparison (default: yolo8)"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (0), IP Webcam URL, or video file path"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto, cpu, cuda (default: auto)"
    )
    
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Save performance metrics to file"
    )
    
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save detected video to results folder"
    )
    
    return parser.parse_args()


def initialize_detectors(args):
    """Initialize object detection models."""
    detectors = {}
    
    if args.model in ["yolo8", "both"]:
        print("\n[Main] Initializing YOLOv8-nano...")
        yolo8 = YOLOv8Detector(
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device
        )
        if yolo8.load_model():
            detectors["yolo8"] = yolo8
        else:
            print("[Main] Warning: Failed to load YOLOv8")
            
    if args.model in ["yolo5", "both"]:
        print("\n[Main] Initializing YOLOv5-small...")
        yolo5 = YOLOv5Detector(
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device
        )
        if yolo5.load_model():
            detectors["yolo5"] = yolo5
        else:
            print("[Main] Warning: Failed to load YOLOv5")
            
    return detectors


def parse_source(source_str):
    """Parse video source argument."""
    # Check if it's an integer (webcam index)
    try:
        return int(source_str)
    except ValueError:
        pass
        
    # Otherwise, it's a URL or file path
    return source_str


def run_detection_loop(
    video_stream: VideoStream,
    detectors: dict,
    config: Config,
    save_metrics: bool = False,
    save_video: bool = False
):
    """
    Main detection loop.
    
    This implements the real-time ML pipeline (Lab 12 concept):
    Frame Capture → Preprocessing → Inference → Visualization
    """
    # Initialize components
    preprocessor = ImageProcessor(target_size=(640, 640), normalize=False, convert_rgb=False)
    visualizer = Visualizer()
    metrics = PerformanceMetrics(window_size=30)
    
    # Get first available model as default
    current_model = list(detectors.keys())[0] if detectors else None
    
    if not current_model:
        print("[Main] Error: No models available")
        return
        
    print(f"\n[Main] Starting detection with {detectors[current_model].model_name}")
    print("[Main] Controls: 1=YOLOv8, 2=YOLOv5, +/- Threshold, Q=Quit")
    print("-" * 50)
    
    # Initialize Video Writer if requested
    video_writer = None
    if save_video:
        os.makedirs(config.get_results_dir(), exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(config.get_results_dir(), f"detection_output_{timestamp}.mp4")
        fps = video_stream.get_fps()
        if fps <= 0: fps = 20.0 # Default if unknown
        res = video_stream.get_resolution()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, res)
        print(f"[Main] Saving video to: {video_path}")
    
    try:
        while True:
            # Start frame timing
            metrics.start_frame()
            
            # Capture frame
            ret, frame = video_stream.read()
            if not ret or frame is None:
                print("[Main] Failed to read frame, attempting reconnection...")
                if not video_stream.connect():
                    print("[Main] Reconnection failed, exiting...")
                    break
                continue
            
            # Preprocessing (Lab 1 concept)
            metrics.start_preprocessing()
            # Note: YOLO handles its own preprocessing, but we still resize for consistency
            processed_frame = cv2.resize(frame, (640, 640)) if frame.shape[:2] != (640, 640) else frame
            metrics.end_preprocessing()
            
            # Detection (Lab 5 concept - multiple models)
            metrics.start_inference()
            detector = detectors[current_model]
            detections = detector.detect(frame)
            inference_time = detector.get_inference_time()
            metrics.end_inference()
            
            # Visualization
            metrics.start_visualization()
            output_frame = visualizer.draw_all(
                frame,
                detections=detections,
                fps=metrics.get_fps(),
                model_name=detector.model_name,
                inference_time_ms=inference_time
            )
            metrics.end_visualization()
            
            # End frame timing
            metrics.end_frame(detector.model_name, len(detections))
            
            # Display
            cv2.imshow(config.window_name, output_frame)
            
            # Save frame if requested
            if video_writer:
                video_writer.write(output_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n[Main] Quitting...")
                break
                
            elif key == ord('1') and "yolo8" in detectors:
                current_model = "yolo8"
                print(f"\n[Main] Switched to {detectors[current_model].model_name}")
                
            elif key == ord('2') and "yolo5" in detectors:
                current_model = "yolo5"
                print(f"\n[Main] Switched to {detectors[current_model].model_name}")
                
            elif key == ord('+') or key == ord('='):
                new_conf = min(0.95, detector.confidence_threshold + 0.05)
                detector.set_confidence_threshold(new_conf)
                
            elif key == ord('-'):
                new_conf = max(0.05, detector.confidence_threshold - 0.05)
                detector.set_confidence_threshold(new_conf)
                
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
        
    finally:
        # Print performance summary (Lab 6 concept - Model Comparison)
        metrics.print_summary()
        
        if save_metrics:
            metrics_path = os.path.join(config.get_results_dir(), "performance_metrics.json")
            os.makedirs(config.get_results_dir(), exist_ok=True)
            metrics.save_metrics(metrics_path)
            
        if video_writer:
            video_writer.release()
            print(f"[Main] Video saved successfully.")
            
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    print("=" * 60)
    print("  Real-time Multi-Model Object Classifier")
    print("  Lab-14: Complex Computing Activity")
    print("=" * 60)
    
    # Parse arguments
    args = parse_args()
    
    print(f"\n[Main] Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Source: {args.source}")
    print(f"  Confidence: {args.conf}")
    print(f"  Device: {args.device}")
    
    # Initialize configuration
    config = Config()
    
    # Initialize detectors
    detectors = initialize_detectors(args)
    
    if not detectors:
        print("\n[Main] Error: Failed to initialize any models")
        sys.exit(1)
        
    # Parse video source
    source = parse_source(args.source)
    
    # Initialize video stream
    print(f"\n[Main] Connecting to video source: {source}")
    video_stream = VideoStream(source=source)
    
    if not video_stream.connect():
        print("[Main] Error: Failed to connect to video source")
        print("\nTips:")
        print("  - For webcam: use --source 0")
        print("  - For IP Webcam: use --source 'http://IP:8080/video'")
        print("  - For video file: use --source 'path/to/video.mp4'")
        sys.exit(1)
        
    print(f"[Main] Connected! Resolution: {video_stream.get_resolution()}")
    
    # Run detection loop
    try:
        run_detection_loop(
            video_stream=video_stream,
            detectors=detectors,
            config=config,
            save_metrics=args.save_metrics,
            save_video=args.save_video
        )
    finally:
        video_stream.release()
        print("\n[Main] Shutdown complete")


if __name__ == "__main__":
    main()
