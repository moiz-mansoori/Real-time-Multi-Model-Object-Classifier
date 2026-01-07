"""
Visualization Module
====================
Drawing bounding boxes, labels, and overlays for object detection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from ..detection.base_detector import Detection
from .config import get_class_color, VisualizationConfig


class Visualizer:
    """
    Visualization handler for object detection results.
    
    Features:
    - Bounding box drawing with class-specific colors
    - Label and confidence display
    - FPS overlay
    - Model name display
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration (uses defaults if None)
        """
        self.config = config or VisualizationConfig()
        
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        use_class_colors: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image: Input image (BGR format)
            detections: List of Detection objects
            use_class_colors: Whether to use class-specific colors
            
        Returns:
            Image with drawn detections
        """
        output = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Get color
            if use_class_colors:
                color = get_class_color(detection.class_name)
            else:
                color = self.config.bbox_color
                
            # Draw bounding box
            cv2.rectangle(
                output,
                (x1, y1), (x2, y2),
                color,
                self.config.bbox_thickness
            )
            
            # Prepare label text
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.config.font_thickness
            )
            
            # Draw label background
            cv2.rectangle(
                output,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 5, y1),
                color,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                output,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.config.text_color,
                self.config.font_thickness
            )
            
        return output
    
    def draw_fps(
        self,
        image: np.ndarray,
        fps: float,
        position: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Draw FPS counter on the image.
        
        Args:
            image: Input image
            fps: Current FPS value
            position: Position for FPS text (uses config default if None)
            
        Returns:
            Image with FPS overlay
        """
        if not self.config.show_fps:
            return image
            
        output = image.copy()
        pos = position or self.config.fps_position
        
        fps_text = f"FPS: {fps:.1f}"
        
        # Draw background
        (text_width, text_height), _ = cv2.getTextSize(
            fps_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )
        
        cv2.rectangle(
            output,
            (pos[0] - 5, pos[1] - text_height - 10),
            (pos[0] + text_width + 10, pos[1] + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw FPS text
        cv2.putText(
            output,
            fps_text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.config.fps_color,
            2
        )
        
        return output
    
    def draw_model_name(
        self,
        image: np.ndarray,
        model_name: str,
        position: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Draw current model name on the image.
        
        Args:
            image: Input image
            model_name: Name of the current model
            position: Position for model name (uses config default if None)
            
        Returns:
            Image with model name overlay
        """
        if not self.config.show_model_name:
            return image
            
        output = image.copy()
        pos = position or self.config.model_name_position
        
        # Draw background
        (text_width, text_height), _ = cv2.getTextSize(
            model_name,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )
        
        cv2.rectangle(
            output,
            (pos[0] - 5, pos[1] - text_height - 10),
            (pos[0] + text_width + 10, pos[1] + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw model name
        cv2.putText(
            output,
            model_name,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),  # Yellow
            2
        )
        
        return output
    
    def draw_info_panel(
        self,
        image: np.ndarray,
        fps: float,
        model_name: str,
        detection_count: int,
        inference_time_ms: float
    ) -> np.ndarray:
        """
        Draw an information panel with all metrics.
        
        Args:
            image: Input image
            fps: Current FPS
            model_name: Active model name
            detection_count: Number of detections
            inference_time_ms: Inference time in milliseconds
            
        Returns:
            Image with info panel
        """
        output = image.copy()
        
        # Panel settings
        panel_x = 10
        panel_y = 10
        line_height = 25
        
        info_lines = [
            f"Model: {model_name}",
            f"FPS: {fps:.1f}",
            f"Latency: {inference_time_ms:.1f} ms",
            f"Detections: {detection_count}",
            "",
            "Controls:",
            "1 - YOLOv8 | 2 - YOLOv5 | Q - Quit"
        ]
        
        # Calculate panel size
        max_width = 0
        for line in info_lines:
            (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            max_width = max(max_width, w)
            
        panel_height = len(info_lines) * line_height + 20
        panel_width = max_width + 20
        
        # Draw semi-transparent panel background
        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (30, 30, 30),
            -1
        )
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        # Draw border
        cv2.rectangle(
            output,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (100, 100, 100),
            1
        )
        
        # Draw text lines
        for i, line in enumerate(info_lines):
            color = (0, 255, 0) if i < 4 else (200, 200, 200)
            cv2.putText(
                output,
                line,
                (panel_x + 10, panel_y + 20 + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
            
        return output
    
    def draw_object_counts(
        self,
        image: np.ndarray,
        detections: List[Detection]
    ) -> np.ndarray:
        """
        Draw an object counting dashboard on the right side of the image.
        
        Shows count of each detected object type with visual bars.
        
        Args:
            image: Input image
            detections: List of Detection objects
            
        Returns:
            Image with object counts panel
        """
        if not detections:
            return image
            
        output = image.copy()
        h, w = output.shape[:2]
        
        # Count objects by class
        class_counts = {}
        for det in detections:
            class_name = det.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Sort by count (descending)
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top 8 classes
        sorted_counts = sorted_counts[:8]
        
        if not sorted_counts:
            return output
        
        # Panel settings (right side)
        panel_width = 200
        panel_x = w - panel_width - 10
        panel_y = 10
        line_height = 30
        bar_max_width = 100
        
        # Calculate panel height
        panel_height = len(sorted_counts) * line_height + 50
        
        # Draw semi-transparent panel background
        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (30, 30, 30),
            -1
        )
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        # Draw border
        cv2.rectangle(
            output,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 200, 255),  # Orange border
            2
        )
        
        # Draw title
        cv2.putText(
            output,
            "OBJECT COUNTS",
            (panel_x + 10, panel_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),  # Orange
            2
        )
        
        # Draw counts with bars
        max_count = max(count for _, count in sorted_counts)
        
        for i, (class_name, count) in enumerate(sorted_counts):
            y_pos = panel_y + 45 + i * line_height
            
            # Get class color
            color = get_class_color(class_name)
            
            # Draw class name and count
            text = f"{class_name}: {count}"
            cv2.putText(
                output,
                text,
                (panel_x + 10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )
            
            # Draw progress bar
            bar_width = int((count / max_count) * bar_max_width)
            bar_y = y_pos + 5
            cv2.rectangle(
                output,
                (panel_x + 10, bar_y),
                (panel_x + 10 + bar_width, bar_y + 8),
                color,
                -1
            )
            # Bar border
            cv2.rectangle(
                output,
                (panel_x + 10, bar_y),
                (panel_x + 10 + bar_max_width, bar_y + 8),
                (100, 100, 100),
                1
            )
        
        return output
    
    def draw_all(
        self,
        image: np.ndarray,
        detections: List[Detection],
        fps: float,
        model_name: str,
        inference_time_ms: float
    ) -> np.ndarray:
        """
        Draw all visualizations on the image.
        
        Args:
            image: Input image
            detections: List of Detection objects
            fps: Current FPS
            model_name: Active model name
            inference_time_ms: Inference time in milliseconds
            
        Returns:
            Fully annotated image
        """
        output = self.draw_detections(image, detections)
        output = self.draw_info_panel(
            output,
            fps=fps,
            model_name=model_name,
            detection_count=len(detections),
            inference_time_ms=inference_time_ms
        )
        # Add object counting dashboard (right side)
        output = self.draw_object_counts(output, detections)
        return output


def draw_comparison_view(
    image1: np.ndarray,
    image2: np.ndarray,
    label1: str = "Model 1",
    label2: str = "Model 2"
) -> np.ndarray:
    """
    Create a side-by-side comparison view of two model outputs.
    
    Args:
        image1: First model output
        image2: Second model output
        label1: Label for first model
        label2: Label for second model
        
    Returns:
        Combined comparison image
    """
    # Ensure same height
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    max_h = max(h1, h2)
    
    if h1 < max_h:
        image1 = cv2.copyMakeBorder(image1, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT)
    if h2 < max_h:
        image2 = cv2.copyMakeBorder(image2, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT)
        
    # Concatenate horizontally
    combined = np.hstack([image1, image2])
    
    # Add labels
    cv2.putText(combined, label1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, label2, (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return combined


if __name__ == "__main__":
    # Test visualization
    print("Testing Visualizer...")
    
    # Create a sample image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (50, 50, 50)  # Gray background
    
    # Create sample detections
    detections = [
        Detection(bbox=(100, 100, 200, 250), class_id=0, class_name="person", confidence=0.92),
        Detection(bbox=(300, 150, 400, 300), class_id=39, class_name="bottle", confidence=0.85),
    ]
    
    visualizer = Visualizer()
    
    output = visualizer.draw_all(
        test_image,
        detections=detections,
        fps=30.5,
        model_name="YOLOv8-nano",
        inference_time_ms=23.5
    )
    
    cv2.imshow("Visualization Test", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
