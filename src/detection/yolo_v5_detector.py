"""
YOLOv5 Detector Module
======================
YOLOv5-small implementation using PyTorch Hub.

Lab Concept Integration:
- Lab 5 (Ensemble/Multiple Models): Second model for comparison
- Lab 6 (Model Comparison): Comparable with YOLOv8 for performance analysis
"""

import time
from typing import List, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[Warning] PyTorch not installed. Run: pip install torch")

from .base_detector import BaseDetector, Detection, COCO_CLASSES


class YOLOv5Detector(BaseDetector):
    """
    YOLOv5-small object detector using PyTorch Hub.
    
    Features:
    - Lightweight and well-established
    - Pretrained on COCO dataset (80 classes)
    - Good balance of speed and accuracy
    """
    
    def __init__(
        self,
        model_name: str = "yolov5s",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto"
    ):
        """
        Initialize YOLOv5-small detector.
        
        Args:
            model_name: Model variant ('yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
            confidence_threshold: Minimum detection confidence
            iou_threshold: IoU threshold for NMS
            device: Device for inference ('auto', 'cpu', 'cuda')
        """
        super().__init__(
            model_name="YOLOv5-small",
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=device
        )
        self.model_variant = model_name
        self.class_names = COCO_CLASSES
        self.last_inference_time = 0.0
        
    def load_model(self) -> bool:
        """
        Load YOLOv5 model from PyTorch Hub.
        
        Returns:
            True if loaded successfully
        """
        if not TORCH_AVAILABLE:
            print("[YOLOv5] Error: PyTorch not available")
            return False
            
        try:
            print(f"[YOLOv5] Loading model: {self.model_variant} from PyTorch Hub...")
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
                
            # Load model from PyTorch Hub
            self.model = torch.hub.load(
                'ultralytics/yolov5',
                self.model_variant,
                pretrained=True,
                trust_repo=True
            )
            
            # Configure model
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            self.model.to(device)
            
            self.device = device
            self.is_loaded = True
            print(f"[YOLOv5] Model loaded successfully on device: {device}")
            return True
            
        except Exception as e:
            print(f"[YOLOv5] Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Perform object detection on an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of Detection objects
        """
        if not self.is_loaded or self.model is None:
            print("[YOLOv5] Model not loaded. Call load_model() first.")
            return []
            
        try:
            start_time = time.perf_counter()
            
            # YOLOv5 expects RGB images
            import cv2
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(rgb_image)
            
            self.last_inference_time = (time.perf_counter() - start_time) * 1000  # ms
            
            # Parse results
            detections = []
            
            # Get predictions as pandas DataFrame
            predictions = results.pandas().xyxy[0]
            
            for _, row in predictions.iterrows():
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                conf = float(row['confidence'])
                cls_id = int(row['class'])
                cls_name = row['name']
                
                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf
                )
                detections.append(detection)
                
            return detections
            
        except Exception as e:
            print(f"[YOLOv5] Detection error: {e}")
            return []
    
    def get_inference_time(self) -> float:
        """Get the last inference time in milliseconds."""
        return self.last_inference_time
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold."""
        super().set_confidence_threshold(threshold)
        if self.model is not None:
            self.model.conf = threshold
    
    def set_iou_threshold(self, threshold: float) -> None:
        """Update IoU threshold."""
        super().set_iou_threshold(threshold)
        if self.model is not None:
            self.model.iou = threshold
    
    def get_model_info(self) -> dict:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            "model_variant": self.model_variant,
            "architecture": "YOLOv5",
            "source": "PyTorch Hub",
            "last_inference_time_ms": self.last_inference_time
        })
        return info


if __name__ == "__main__":
    # Test YOLOv5 detector
    print("Testing YOLOv5 Detector...")
    
    import cv2
    
    detector = YOLOv5Detector()
    
    if detector.load_model():
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detections = detector.detect(test_image)
        
        print(f"Detections: {len(detections)}")
        print(f"Inference time: {detector.get_inference_time():.2f} ms")
        print(f"Model info: {detector.get_model_info()}")
    else:
        print("Failed to load model")
