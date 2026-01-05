"""
Base Detector Module
====================
Abstract base class for object detection models.

Lab Concept Integration:
- Lab 5 (Ensemble/Multiple Models): Common interface for multiple model implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    """
    Data class representing a single detection result.
    
    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        class_id: Class index from the model
        class_name: Human-readable class name
        confidence: Detection confidence score (0-1)
    """
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    
    def get_center(self) -> Tuple[int, int]:
        """Get the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def get_area(self) -> int:
        """Get the area of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def __repr__(self) -> str:
        return f"Detection({self.class_name}: {self.confidence:.2f} @ {self.bbox})"


class BaseDetector(ABC):
    """
    Abstract base class for object detection models.
    
    This class defines the common interface that all detector implementations
    must follow. This design allows easy comparison between different models
    (Lab 5 & Lab 6 concepts).
    """
    
    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto"
    ):
        """
        Initialize base detector.
        
        Args:
            model_name: Name/identifier for the model
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference ('auto', 'cpu', 'cuda', 'cuda:0')
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.class_names: List[str] = []
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the model weights.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Perform object detection on an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            List of Detection objects
        """
        pass
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update the confidence threshold.
        
        This allows for threshold tuning (Lab 6 concept: hyperparameter optimization).
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            print(f"[{self.model_name}] Confidence threshold set to {threshold:.2f}")
        else:
            raise ValueError("Confidence threshold must be between 0 and 1")
    
    def set_iou_threshold(self, threshold: float) -> None:
        """
        Update the IoU threshold for NMS.
        
        This allows for NMS tuning (Lab 6 concept: hyperparameter optimization).
        """
        if 0.0 <= threshold <= 1.0:
            self.iou_threshold = threshold
            print(f"[{self.model_name}] IoU threshold set to {threshold:.2f}")
        else:
            raise ValueError("IoU threshold must be between 0 and 1")
    
    def get_class_names(self) -> List[str]:
        """Get the list of class names the model can detect."""
        return self.class_names
    
    def get_model_info(self) -> dict:
        """Get model information for reporting."""
        return {
            "name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "num_classes": len(self.class_names),
            "is_loaded": self.is_loaded
        }
    
    def filter_detections(
        self,
        detections: List[Detection],
        target_classes: Optional[List[str]] = None,
        min_confidence: Optional[float] = None
    ) -> List[Detection]:
        """
        Filter detections by class names and/or confidence.
        
        Args:
            detections: List of Detection objects
            target_classes: Optional list of class names to keep
            min_confidence: Optional minimum confidence threshold
            
        Returns:
            Filtered list of Detection objects
        """
        filtered = detections
        
        if target_classes:
            filtered = [d for d in filtered if d.class_name in target_classes]
            
        if min_confidence:
            filtered = [d for d in filtered if d.confidence >= min_confidence]
            
        return filtered
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, loaded={self.is_loaded})"


# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
