"""
Performance Metrics Module
==========================
FPS and latency measurement for model comparison.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    timestamp: float
    inference_time_ms: float
    preprocessing_time_ms: float
    visualization_time_ms: float
    total_time_ms: float
    detection_count: int
    model_name: str


class PerformanceMetrics:
    """
    Performance metrics tracker for real-time object detection.
    
    Tracks:
    - FPS (frames per second)
    - Inference latency
    - Preprocessing time
    - Total pipeline time
    - Detection counts
    """
    
    def __init__(
        self,
        window_size: int = 30,
        log_to_file: bool = False,
        log_path: Optional[str] = None
    ):
        """
        Initialize performance metrics tracker.
        
        Args:
            window_size: Number of frames for rolling average FPS
            log_to_file: Whether to log metrics to file
            log_path: Path for metrics log file
        """
        self.window_size = window_size
        self.log_to_file = log_to_file
        self.log_path = log_path
        
        # Rolling window for FPS calculation
        self.frame_times: deque = deque(maxlen=window_size)
        
        # Metrics storage per model
        self.model_metrics: Dict[str, List[FrameMetrics]] = {}
        
        # Current frame timers
        self._frame_start: float = 0
        self._preprocess_start: float = 0
        self._inference_start: float = 0
        self._viz_start: float = 0
        
        # Current frame measurements
        self._current_preprocess_time: float = 0
        self._current_inference_time: float = 0
        self._current_viz_time: float = 0
        
        # Total frame counter
        self.total_frames: int = 0
        
    def start_frame(self) -> None:
        """Mark the start of a new frame."""
        self._frame_start = time.perf_counter()
        
    def start_preprocessing(self) -> None:
        """Mark the start of preprocessing."""
        self._preprocess_start = time.perf_counter()
        
    def end_preprocessing(self) -> None:
        """Mark the end of preprocessing."""
        self._current_preprocess_time = (time.perf_counter() - self._preprocess_start) * 1000
        
    def start_inference(self) -> None:
        """Mark the start of inference."""
        self._inference_start = time.perf_counter()
        
    def end_inference(self) -> None:
        """Mark the end of inference."""
        self._current_inference_time = (time.perf_counter() - self._inference_start) * 1000
        
    def start_visualization(self) -> None:
        """Mark the start of visualization."""
        self._viz_start = time.perf_counter()
        
    def end_visualization(self) -> None:
        """Mark the end of visualization."""
        self._current_viz_time = (time.perf_counter() - self._viz_start) * 1000
        
    def end_frame(self, model_name: str, detection_count: int) -> FrameMetrics:
        """
        Mark the end of a frame and record metrics.
        
        Args:
            model_name: Name of the model used
            detection_count: Number of detections in this frame
            
        Returns:
            FrameMetrics for this frame
        """
        frame_end = time.perf_counter()
        total_time = (frame_end - self._frame_start) * 1000
        
        # Record frame time for FPS calculation
        self.frame_times.append(frame_end)
        self.total_frames += 1
        
        # Create frame metrics
        metrics = FrameMetrics(
            timestamp=frame_end,
            inference_time_ms=self._current_inference_time,
            preprocessing_time_ms=self._current_preprocess_time,
            visualization_time_ms=self._current_viz_time,
            total_time_ms=total_time,
            detection_count=detection_count,
            model_name=model_name
        )
        
        # Store metrics by model
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = []
        self.model_metrics[model_name].append(metrics)
        
        # Reset current frame measurements
        self._current_preprocess_time = 0
        self._current_inference_time = 0
        self._current_viz_time = 0
        
        return metrics
    
    def get_fps(self) -> float:
        """
        Calculate current FPS using rolling average.
        
        Returns:
            Frames per second (float)
        """
        if len(self.frame_times) < 2:
            return 0.0
            
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff == 0:
            return 0.0
            
        return (len(self.frame_times) - 1) / time_diff
    
    def get_average_inference_time(self, model_name: Optional[str] = None) -> float:
        """
        Get average inference time.
        
        Args:
            model_name: Optional model name filter
            
        Returns:
            Average inference time in milliseconds
        """
        metrics_list = []
        
        if model_name:
            metrics_list = self.model_metrics.get(model_name, [])
        else:
            for model_metrics in self.model_metrics.values():
                metrics_list.extend(model_metrics)
                
        if not metrics_list:
            return 0.0
            
        return sum(m.inference_time_ms for m in metrics_list) / len(metrics_list)
    
    def get_model_comparison(self) -> Dict[str, dict]:
        """
        Get comparison metrics for all models.
        
        This is key for Lab 6: Model Comparison & Optimization
        
        Returns:
            Dictionary with comparison metrics per model
        """
        comparison = {}
        
        for model_name, metrics_list in self.model_metrics.items():
            if not metrics_list:
                continue
                
            inference_times = [m.inference_time_ms for m in metrics_list]
            total_times = [m.total_time_ms for m in metrics_list]
            detection_counts = [m.detection_count for m in metrics_list]
            
            comparison[model_name] = {
                "total_frames": len(metrics_list),
                "avg_inference_time_ms": sum(inference_times) / len(inference_times),
                "min_inference_time_ms": min(inference_times),
                "max_inference_time_ms": max(inference_times),
                "avg_total_time_ms": sum(total_times) / len(total_times),
                "avg_detections_per_frame": sum(detection_counts) / len(detection_counts),
                "theoretical_max_fps": 1000 / (sum(total_times) / len(total_times)) if total_times else 0
            }
            
        return comparison
    
    def print_summary(self) -> None:
        """Print a summary of performance metrics."""
        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS SUMMARY")
        print("=" * 60)
        
        comparison = self.get_model_comparison()
        
        for model_name, metrics in comparison.items():
            print(f"\n📊 {model_name}")
            print(f"   Frames processed: {metrics['total_frames']}")
            print(f"   Avg inference time: {metrics['avg_inference_time_ms']:.2f} ms")
            print(f"   Min/Max inference: {metrics['min_inference_time_ms']:.2f} / {metrics['max_inference_time_ms']:.2f} ms")
            print(f"   Avg total time: {metrics['avg_total_time_ms']:.2f} ms")
            print(f"   Theoretical max FPS: {metrics['theoretical_max_fps']:.1f}")
            print(f"   Avg detections/frame: {metrics['avg_detections_per_frame']:.1f}")
            
        print("\n" + "=" * 60)
    
    def save_metrics(self, filepath: str) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            filepath: Path to save the metrics
        """
        data = {
            "total_frames": self.total_frames,
            "model_comparison": self.get_model_comparison(),
            "detailed_metrics": {}
        }
        
        # Add detailed metrics (last 100 frames per model)
        for model_name, metrics_list in self.model_metrics.items():
            data["detailed_metrics"][model_name] = [
                {
                    "inference_time_ms": m.inference_time_ms,
                    "total_time_ms": m.total_time_ms,
                    "detection_count": m.detection_count
                }
                for m in metrics_list[-100:]
            ]
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"[Metrics] Saved to {filepath}")
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.frame_times.clear()
        self.model_metrics.clear()
        self.total_frames = 0


if __name__ == "__main__":
    # Test performance metrics
    print("Testing PerformanceMetrics...")
    
    metrics = PerformanceMetrics(window_size=10)
    
    # Simulate some frames
    for i in range(50):
        metrics.start_frame()
        
        metrics.start_preprocessing()
        time.sleep(0.002)  # 2ms preprocessing
        metrics.end_preprocessing()
        
        metrics.start_inference()
        time.sleep(0.02)  # 20ms inference
        metrics.end_inference()
        
        metrics.start_visualization()
        time.sleep(0.001)  # 1ms visualization
        metrics.end_visualization()
        
        model = "YOLOv8" if i % 2 == 0 else "YOLOv5"
        metrics.end_frame(model, detection_count=5)
        
    print(f"Current FPS: {metrics.get_fps():.1f}")
    metrics.print_summary()
