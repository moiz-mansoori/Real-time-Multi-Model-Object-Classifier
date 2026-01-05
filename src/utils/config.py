"""
Configuration Module
====================
Central configuration for the Real-time Multi-Model Object Classifier.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CameraConfig:
    """Camera/video source configuration."""
    default_source: int = 0  # Default webcam
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    ip_webcam_port: int = 8080


@dataclass
class ModelConfig:
    """Model configuration."""
    # YOLOv8 settings
    yolov8_model: str = "yolov8n.pt"
    yolov8_name: str = "YOLOv8-nano"
    
    # YOLOv5 settings
    yolov5_model: str = "yolov5s"
    yolov5_name: str = "YOLOv5-small"
    
    # Detection settings
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    
    # Device settings
    device: str = "auto"  # 'auto', 'cpu', 'cuda', 'cuda:0'


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    # Bounding box settings
    bbox_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2
    
    # Colors (BGR format for OpenCV)
    bbox_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    text_color: Tuple[int, int, int] = (255, 255, 255)  # White
    text_bg_color: Tuple[int, int, int] = (0, 0, 0)  # Black
    
    # FPS display
    show_fps: bool = True
    fps_position: Tuple[int, int] = (10, 30)
    fps_color: Tuple[int, int, int] = (0, 255, 0)
    
    # Model name display
    show_model_name: bool = True
    model_name_position: Tuple[int, int] = (10, 60)


@dataclass
class PerformanceConfig:
    """Performance measurement settings."""
    fps_smoothing_window: int = 30  # Number of frames for rolling average
    log_metrics: bool = True
    metrics_log_interval: int = 100  # Log every N frames


@dataclass
class Config:
    """Main configuration class containing all settings."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Project paths
    project_root: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Target classes (common objects for demo)
    target_classes: List[str] = field(default_factory=lambda: [
        "person", "bottle", "cup", "cell phone", "laptop",
        "chair", "keyboard", "mouse", "book", "remote"
    ])
    
    # Window settings
    window_name: str = "Real-time Multi-Model Object Classifier"
    
    def get_models_dir(self) -> str:
        """Get the models directory path."""
        return os.path.join(self.project_root, "models")
    
    def get_results_dir(self) -> str:
        """Get the results directory path."""
        return os.path.join(self.project_root, "results")
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(self.get_models_dir(), exist_ok=True)
        os.makedirs(self.get_results_dir(), exist_ok=True)


# Color palette for different classes (for visual variety)
CLASS_COLORS = {
    "person": (255, 0, 0),      # Blue
    "bottle": (0, 255, 0),      # Green
    "cup": (0, 0, 255),         # Red
    "cell phone": (255, 255, 0), # Cyan
    "laptop": (255, 0, 255),    # Magenta
    "chair": (0, 255, 255),     # Yellow
    "keyboard": (128, 0, 128),  # Purple
    "mouse": (0, 128, 128),     # Olive
    "book": (128, 128, 0),      # Teal
    "remote": (128, 0, 0),      # Navy
}


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """
    Get color for a specific class.
    
    Args:
        class_name: Name of the detected class
        
    Returns:
        BGR color tuple
    """
    return CLASS_COLORS.get(class_name, (0, 255, 0))  # Default: green


# Default configuration instance
DEFAULT_CONFIG = Config()


if __name__ == "__main__":
    # Test configuration
    config = Config()
    print(f"Project root: {config.project_root}")
    print(f"Models dir: {config.get_models_dir()}")
    print(f"Camera resolution: {config.camera.resolution}")
    print(f"Confidence threshold: {config.model.confidence_threshold}")
    print(f"Target classes: {config.target_classes}")
