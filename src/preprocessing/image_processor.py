"""
Image Processor Module
======================
Handles image preprocessing for object detection models.

Lab Concept Integration:
- Lab 1 (Data Preprocessing): Image resizing, normalization, color conversion
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImageProcessor:
    """
    Image preprocessing pipeline for object detection.
    
    Implements:
    - Resizing to model input dimensions
    - Pixel normalization (0-1 range)
    - Color space conversion (BGR to RGB)
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        normalize: bool = True,
        convert_rgb: bool = True
    ):
        """
        Initialize image processor.
        
        Args:
            target_size: Target dimensions (width, height) for model input
            normalize: Whether to normalize pixel values to [0, 1]
            convert_rgb: Whether to convert BGR to RGB
        """
        self.target_size = target_size
        self.normalize = normalize
        self.convert_rgb = convert_rgb
        
    def preprocess(
        self,
        image: np.ndarray,
        keep_aspect_ratio: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply full preprocessing pipeline to an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            keep_aspect_ratio: Whether to maintain aspect ratio when resizing
            
        Returns:
            Tuple of (preprocessed_image, preprocessing_info)
            preprocessing_info contains original dimensions and padding info
        """
        if image is None:
            raise ValueError("Input image is None")
            
        original_height, original_width = image.shape[:2]
        
        # Store preprocessing info for later use
        preprocess_info = {
            'original_size': (original_width, original_height),
            'target_size': self.target_size,
            'padding': (0, 0, 0, 0)  # top, bottom, left, right
        }
        
        # Step 1: Resize
        if keep_aspect_ratio:
            processed, padding = self._resize_with_padding(image)
            preprocess_info['padding'] = padding
        else:
            processed = self._resize(image)
            
        # Step 2: Color conversion (BGR to RGB)
        if self.convert_rgb:
            processed = self._bgr_to_rgb(processed)
            
        # Step 3: Normalize pixel values
        if self.normalize:
            processed = self._normalize(processed)
            
        return processed, preprocess_info
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target dimensions.
        
        This is a key preprocessing step (Lab 1 concept):
        - Ensures consistent input size for the model
        - Uses bilinear interpolation for quality
        """
        return cv2.resize(
            image,
            self.target_size,
            interpolation=cv2.INTER_LINEAR
        )
    
    def _resize_with_padding(
        self,
        image: np.ndarray,
        pad_color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Resize image maintaining aspect ratio with padding.
        
        Args:
            image: Input image
            pad_color: Color for padding (default: gray)
            
        Returns:
            Tuple of (resized_padded_image, padding_info)
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # Apply padding
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=pad_color
        )
        
        return padded, (top, bottom, left, right)
    
    def _bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR to RGB color space.
        
        OpenCV loads images in BGR format, but most ML models expect RGB.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] range.
        
        This is a key preprocessing step (Lab 1 concept):
        - Neural networks work better with normalized inputs
        - Ensures consistent scale across different images
        """
        return image.astype(np.float32) / 255.0
    
    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        Reverse normalization for visualization.
        
        Args:
            image: Normalized image with values in [0, 1]
            
        Returns:
            Image with pixel values in [0, 255]
        """
        return (image * 255).astype(np.uint8)
    
    def rgb_to_bgr(self, image: np.ndarray) -> np.ndarray:
        """
        Convert RGB back to BGR for OpenCV visualization.
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Resize and pad image for YOLO models (letterbox transformation).
    
    This function is commonly used in YOLO preprocessing pipelines.
    
    Args:
        img: Input image
        new_shape: Target shape
        color: Padding color
        auto: Minimum rectangle padding
        scale_fill: Stretch to fill
        scaleup: Allow scaling up
        stride: Model stride for padding alignment
        
    Returns:
        Tuple of (processed_image, ratio, padding)
    """
    shape = img.shape[:2]  # current shape [height, width]
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
        
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
        
    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, ratio, (dw, dh)


if __name__ == "__main__":
    # Test the image processor
    print("Testing ImageProcessor...")
    
    # Create a sample image
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    processor = ImageProcessor(target_size=(640, 640))
    
    processed, info = processor.preprocess(sample_image)
    
    print(f"Original size: {info['original_size']}")
    print(f"Target size: {info['target_size']}")
    print(f"Processed shape: {processed.shape}")
    print(f"Processed dtype: {processed.dtype}")
    print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")
