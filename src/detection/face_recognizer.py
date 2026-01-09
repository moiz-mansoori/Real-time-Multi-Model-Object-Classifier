"""
Face Recognition Module
=======================
Simple face recognition using OpenCV's face detection and feature matching.
Alerts when a known face is detected in the video stream.

This module avoids dlib/face_recognition dependency issues on Windows.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class FaceRecognizer:
    """
    Face recognizer using OpenCV's face detection and histogram comparison.
    
    Features:
    - Load known faces from a folder
    - Detect faces in video frames
    - Match detected faces against known faces
    - Trigger alerts for recognized faces
    """
    
    def __init__(
        self,
        known_faces_dir: str = "known_faces",
        similarity_threshold: float = 0.6,
        alert_cooldown: float = 3.0
    ):
        """
        Initialize face recognizer.
        
        Args:
            known_faces_dir: Directory containing known face images
            similarity_threshold: Threshold for face matching (0-1, higher = stricter)
            alert_cooldown: Seconds between alerts for same person
        """
        self.known_faces_dir = known_faces_dir
        self.similarity_threshold = similarity_threshold
        self.alert_cooldown = alert_cooldown
        
        # Known face data
        self.known_faces = []  # List of (name, face_encoding)
        self.known_names = []
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Alert state
        self.last_alert_time = {}
        self.alert_triggered = False
        self.alert_message = ""
        
    def load_known_faces(self) -> int:
        """
        Load known faces from the directory.
        
        Expected structure:
        known_faces/
        ├── person_name/
        │   ├── image1.jpg
        │   └── image2.jpg
        
        Or simply:
        known_faces/
        ├── image/
        │   ├── photo1.jpg
        │   └── photo2.jpg
        
        Returns:
            Number of faces loaded
        """
        if not os.path.exists(self.known_faces_dir):
            print(f"[FaceRecognizer] Directory not found: {self.known_faces_dir}")
            return 0
            
        count = 0
        
        for person_dir in os.listdir(self.known_faces_dir):
            person_path = os.path.join(self.known_faces_dir, person_dir)
            
            if os.path.isdir(person_path):
                person_name = person_dir
                
                for img_file in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_file)
                    
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        encoding = self._encode_face_from_file(img_path)
                        
                        if encoding is not None:
                            self.known_faces.append(encoding)
                            self.known_names.append(person_name)
                            count += 1
                            print(f"[FaceRecognizer] Loaded face: {person_name} from {img_file}")
                            
        print(f"[FaceRecognizer] Total faces loaded: {count}")
        return count
    
    def _encode_face_from_file(self, image_path: str) -> Optional[np.ndarray]:
        """
        Create a face encoding from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Face encoding (histogram) or None if no face found
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) == 0:
                # Try with the whole image if no face detected
                return self._compute_face_encoding(image)
                
            # Use the first face found
            x, y, w, h = faces[0]
            face_roi = image[y:y+h, x:x+w]
            
            return self._compute_face_encoding(face_roi)
            
        except Exception as e:
            print(f"[FaceRecognizer] Error encoding face from {image_path}: {e}")
            return None
    
    def _compute_face_encoding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Compute face encoding using color histogram.
        
        Args:
            face_image: Face ROI image
            
        Returns:
            Face encoding vector
        """
        # Resize to standard size
        face_resized = cv2.resize(face_image, (100, 100))
        
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
        
        # Compute histogram
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, 
            [8, 8, 8], [0, 180, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def detect_and_recognize(
        self, 
        frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], str, float, bool]]:
        """
        Detect and recognize faces in a frame.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of (bbox, name, confidence, is_known) tuples
        """
        import time
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        
        results = []
        self.alert_triggered = False
        self.alert_message = ""
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            encoding = self._compute_face_encoding(face_roi)
            
            # Compare with known faces
            best_match = None
            best_score = 0
            
            for i, known_encoding in enumerate(self.known_faces):
                # Compare histograms using correlation
                score = cv2.compareHist(encoding, known_encoding, cv2.HISTCMP_CORREL)
                
                if score > best_score:
                    best_score = score
                    best_match = self.known_names[i]
            
            is_known = best_score > self.similarity_threshold
            name = best_match if is_known else "Unknown"
            
            # Handle alert
            if is_known:
                current_time = time.time()
                last_alert = self.last_alert_time.get(name, 0)
                
                if current_time - last_alert > self.alert_cooldown:
                    self.alert_triggered = True
                    self.alert_message = f"🚨 ALERT: {name} DETECTED!"
                    self.last_alert_time[name] = current_time
                    print(f"\n{'='*50}")
                    print(f"🚨 FACE ALERT: {name} detected! (confidence: {best_score:.2f})")
                    print(f"{'='*50}\n")
            
            results.append(((x, y, x+w, y+h), name, best_score, is_known))
            
        return results
    
    def draw_face_results(
        self,
        image: np.ndarray,
        results: List[Tuple[Tuple[int, int, int, int], str, float, bool]]
    ) -> np.ndarray:
        """
        Draw face detection and recognition results on the image.
        
        Args:
            image: Input image
            results: List of detection results from detect_and_recognize
            
        Returns:
            Image with drawn results
        """
        output = image.copy()
        
        for (bbox, name, confidence, is_known) in results:
            x1, y1, x2, y2 = bbox
            
            # Color: Green for known, Red for unknown
            color = (0, 255, 0) if is_known else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{name}: {confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                output, 
                (x1, y1 - text_h - 10), 
                (x1 + text_w + 5, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                output, label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2
            )
            
        # Draw alert if triggered
        if self.alert_triggered:
            h, w = output.shape[:2]
            
            # Alert banner at top
            cv2.rectangle(output, (0, 0), (w, 50), (0, 0, 255), -1)
            cv2.putText(
                output, self.alert_message,
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2
            )
            
        return output


if __name__ == "__main__":
    # Test face recognizer
    print("Testing Face Recognizer...")
    
    recognizer = FaceRecognizer(known_faces_dir="known_faces")
    count = recognizer.load_known_faces()
    
    print(f"Loaded {count} known faces")
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = recognizer.detect_and_recognize(frame)
        output = recognizer.draw_face_results(frame, results)
        
        cv2.imshow("Face Recognition Test", output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
