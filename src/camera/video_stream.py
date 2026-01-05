"""
Video Stream Module
===================
Handles camera input from IP Webcam, local webcam, or video files.

Lab Concept Integration:
- Lab 12 (Real-Time ML Pipeline): Live camera feed acquisition
"""

import cv2
import time
from typing import Optional, Tuple, Union


class VideoStream:
    """
    Video stream handler for multiple input sources.
    
    Supports:
    - IP Webcam (Android app) via URL
    - Local webcam via device index
    - Video files for testing
    """
    
    def __init__(
        self,
        source: Union[str, int] = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30
    ):
        """
        Initialize video stream.
        
        Args:
            source: Video source - URL string, device index (int), or file path
            resolution: Desired resolution (width, height)
            fps: Desired frames per second
        """
        self.source = source
        self.resolution = resolution
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        
    def connect(self, max_retries: int = 3, retry_delay: float = 2.0) -> bool:
        """
        Establish connection to video source with retry logic.
        
        Args:
            max_retries: Maximum connection attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if connection successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                print(f"[VideoStream] Connecting to source: {self.source} (attempt {attempt + 1}/{max_retries})")
                
                self.cap = cv2.VideoCapture(self.source)
                
                if self.cap.isOpened():
                    # Set resolution
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                    
                    # Verify by reading a test frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.is_connected = True
                        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"[VideoStream] Connected successfully! Resolution: {actual_width}x{actual_height}")
                        return True
                        
            except Exception as e:
                print(f"[VideoStream] Connection error: {e}")
                
            if attempt < max_retries - 1:
                print(f"[VideoStream] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                
        print("[VideoStream] Failed to connect after all retries")
        self.is_connected = False
        return False
    
    def read(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        """
        Read a frame from the video stream.
        
        Returns:
            Tuple of (success, frame) where frame is None if read failed
        """
        if not self.is_connected or self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        
        if not ret:
            self.is_connected = False
            return False, None
            
        return True, frame
    
    def get_fps(self) -> float:
        """Get the actual FPS of the video source."""
        if self.cap is not None:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get the actual resolution of the video source."""
        if self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)
    
    def release(self) -> None:
        """Release the video capture resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        print("[VideoStream] Released video capture")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


def create_ip_webcam_url(ip: str, port: int = 8080) -> str:
    """
    Create IP Webcam URL from IP address and port.
    
    Args:
        ip: IP address of the phone (e.g., "192.168.1.10")
        port: Port number (default: 8080)
        
    Returns:
        Full video stream URL
    """
    return f"http://{ip}:{port}/video"


if __name__ == "__main__":
    # Test with default webcam
    print("Testing VideoStream with default webcam...")
    
    with VideoStream(source=0) as stream:
        if stream.is_connected:
            print(f"Resolution: {stream.get_resolution()}")
            print(f"FPS: {stream.get_fps()}")
            
            # Read and display a few frames
            for i in range(30):
                ret, frame = stream.read()
                if ret:
                    cv2.imshow("Test Frame", frame)
                    if cv2.waitKey(33) & 0xFF == ord('q'):
                        break
                        
            cv2.destroyAllWindows()
        else:
            print("Failed to connect to webcam")
