"""
YOLOv8 Detector Module
======================
YOLOv8-nano implementation using Ultralytics library.

Lab Concept Integration:
- Lab 5 (Ensemble/Multiple Models): One of the multi-model implementations
- Lab 6 (Model Comparison): Comparable with YOLOv5 for performance analysis
"""

import time
from typing import List, Optional
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("[Warning] Ultralytics not installed. Run: pip install ultralytics")

from .base_detector import BaseDetector, Detection, COCO_CLASSES


class YOLOv8Detector(BaseDetector):
    """
    YOLOv8-nano object detector using Ultralytics library.
    
    Features:
    - Lightweight and fast (nano variant)
    - Pretrained on COCO dataset (80 classes)
    - State-of-the-art accuracy for its size
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto"
    ):
        """
        Initialize YOLOv8-nano detector.
        
        Args:
            model_path: Path to model weights or model name (auto-downloads)
            confidence_threshold: Minimum detection confidence
            iou_threshold: IoU threshold for NMS
            device: Device for inference ('auto', 'cpu', 'cuda')
        """
        super().__init__(
            model_name="YOLOv8-nano",
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=device
        )
        self.model_path = model_path
        self.class_names = COCO_CLASSES
        self.last_inference_time = 0.0
        
    def load_model(self) -> bool:
        """
        Load YOLOv8 model weights.
        
        Returns:
            True if loaded successfully
        """
        if not ULTRALYTICS_AVAILABLE:
            print("[YOLOv8] Error: Ultralytics library not available")
            return False
            
        try:
            print(f"[YOLOv8] Loading model from: {self.model_path}")
            
            # Load model (auto-downloads if not present)
            self.model = YOLO(self.model_path)
            
            # Set device
            if self.device == "auto":
                # Ultralytics handles device selection automatically
                pass
            else:
                self.model.to(self.device)
                
            self.is_loaded = True
            print(f"[YOLOv8] Model loaded successfully on device: {self.device}")
            return True
            
        except Exception as e:
            print(f"[YOLOv8] Failed to load model: {e}")
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
            print("[YOLOv8] Model not loaded. Call load_model() first.")
            return []
            
        try:
            start_time = time.perf_counter()
            
            # Run inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            self.last_inference_time = (time.perf_counter() - start_time) * 1000  # ms
            
            # Parse results
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is None:
                    continue
                    
                for i in range(len(boxes)):
                    # Get bounding box
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Get confidence
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    # Get class
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                    
                    detection = Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf
                    )
                    detections.append(detection)
                    
            return detections
            
        except Exception as e:
            print(f"[YOLOv8] Detection error: {e}")
            return []
    
    def get_inference_time(self) -> float:
        """Get the last inference time in milliseconds."""
        return self.last_inference_time
    
    def get_model_info(self) -> dict:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            "model_path": self.model_path,
            "architecture": "YOLOv8",
            "variant": "nano",
            "last_inference_time_ms": self.last_inference_time
        })
        return info


if __name__ == "__main__":
    # Test YOLOv8 detector
    print("Testing YOLOv8 Detector...")
    
    import cv2
    
    detector = YOLOv8Detector()
    
    if detector.load_model():
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detections = detector.detect(test_image)
        
        print(f"Detections: {len(detections)}")
        print(f"Inference time: {detector.get_inference_time():.2f} ms")
        print(f"Model info: {detector.get_model_info()}")
    else:
        print("Failed to load model")
