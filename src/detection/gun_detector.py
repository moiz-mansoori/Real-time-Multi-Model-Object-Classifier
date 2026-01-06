"""
Gun Detector Module
===================
Custom gun detection using pre-trained YOLOv8 model.
For Smart Home Security applications.
"""

import time
from typing import List
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from .base_detector import BaseDetector, Detection


class GunDetector(BaseDetector):
    """
    Gun detector using custom pre-trained YOLOv8 model.
    
    Features:
    - Detects guns/weapons in real-time
    - Pretrained specifically for weapon detection
    """
    
    def __init__(
        self,
        model_path: str = "models/gun_detector.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto"
    ):
        super().__init__(
            model_name="Gun-Detector",
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=device
        )
        self.model_path = model_path
        self.class_names = ["gun", "weapon"]  # May vary based on training
        self.last_inference_time = 0.0
        
    def load_model(self) -> bool:
        if not ULTRALYTICS_AVAILABLE:
            print("[GunDetector] Error: Ultralytics not available")
            return False
            
        try:
            print(f"[GunDetector] Loading model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            print("[GunDetector] Model loaded successfully")
            return True
        except Exception as e:
            print(f"[GunDetector] Failed to load model: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        if not self.is_loaded or self.model is None:
            return []
            
        try:
            start_time = time.perf_counter()
            
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            self.last_inference_time = (time.perf_counter() - start_time) * 1000
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Get class name from model
                    cls_name = result.names.get(cls_id, "gun")
                    
                    detection = Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf
                    )
                    detections.append(detection)
                    
            return detections
        except Exception as e:
            print(f"[GunDetector] Detection error: {e}")
            return []
    
    def get_inference_time(self) -> float:
        return self.last_inference_time
    
    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            "model_path": self.model_path,
            "architecture": "YOLOv8-Gun",
            "last_inference_time_ms": self.last_inference_time
        })
        return info
