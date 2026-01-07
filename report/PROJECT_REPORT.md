# Lab-14: Complex Computing Activity
# Real-time Multi-Model Object Classifier (Mobile-Based)

**Course:** Machine Learning (BSAI-462)  
**Semester:** 7th Semester, 4th Year  
**Batch:** 2022  

**Roll Number:** 22F-BSAI-29,22F-BSAI-32,22F-BSAI-43,22F-BSAI-09  
**Submission Date:** 2026-01-07

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Lab Concepts Integration](#3-lab-concepts-integration)
4. [Dataset Description](#4-dataset-description)
5. [Data Preprocessing](#5-data-preprocessing)
6. [System Architecture](#6-system-architecture)
7. [Model Design](#7-model-design)
8. [Model Evaluation](#8-model-evaluation)
9. [Real-Time Performance Analysis](#9-real-time-performance-analysis)
10. [Execution Environment](#10-execution-environment)
11. [Challenges & Limitations](#11-challenges--limitations)
12. [Ethical Considerations](#12-ethical-considerations)
13. [Conclusion & Future Work](#13-conclusion--future-work)
14. [References](#14-references)

---

## 1. Introduction

This project implements a **Real-time Multi-Model Object Classifier** that uses a mobile phone camera as the video input source and employs multiple machine learning models for object detection. The system demonstrates practical application of various ML concepts learned throughout the course labs.

### Project Objectives
- Build a real-time object detection pipeline using mobile camera input
- Implement and compare multiple detection models (YOLOv8-nano and YOLOv5-small)
- Measure and analyze performance metrics (FPS, latency, detection accuracy)
- Integrate concepts from multiple lab experiments into a cohesive system

### Key Features
- Real-time object detection from IP Webcam video stream
- Multi-model support with runtime switching
- Performance metrics visualization (FPS, latency)
- Bounding boxes with class labels and confidence scores
- Professional code structure following software engineering best practices

---

## 2. Problem Statement

Object detection in real-time video streams is a fundamental computer vision task with applications in surveillance, autonomous vehicles, robotics, and augmented reality. The challenge lies in achieving:

1. **Real-time performance** - Processing frames fast enough for smooth video display (>20 FPS)
2. **Accuracy** - Correctly detecting and classifying objects with high confidence
3. **Multi-model comparison** - Evaluating trade-offs between different model architectures
4. **Resource efficiency** - Running efficiently on available hardware

This project addresses these challenges by implementing a complete pipeline that:
- Captures video from a mobile phone camera via Wi-Fi
- Applies preprocessing optimized for detection models
- Runs inference using multiple YOLO variants
- Visualizes results with performance metrics

---

## 3. Lab Concepts Integration

This project explicitly integrates **four distinct concepts** from the ML Lab Manual:

### 3.1 Lab 1 — Data Preprocessing
**Location:** `src/preprocessing/image_processor.py`

The preprocessing module implements essential image preprocessing steps:

| Technique | Implementation | Purpose |
|-----------|----------------|---------|
| Image Resizing | Resize to 640×640 | Consistent input size for models |
| Normalization | Scale pixels to [0,1] | Neural network optimization |
| Color Conversion | BGR → RGB | Model compatibility |

```python
# Example from image_processor.py
def _normalize(self, image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1] range."""
    return image.astype(np.float32) / 255.0
```

### 3.2 Lab 5 — Ensemble/Multiple Models
**Location:** `src/detection/yolo_v8_detector.py`, `src/detection/yolo_v5_detector.py`

The system implements multiple detection models:

| Model | Architecture | Characteristics |
|-------|--------------|-----------------|
| YOLOv8-nano | Ultralytics YOLOv8 | Newest, fastest inference |
| YOLOv5-small | PyTorch Hub | Well-established, balanced |

Both models share a common interface (`BaseDetector`) enabling seamless switching and comparison.

### 3.3 Lab 6 — Model Comparison & Optimization
**Location:** `src/utils/performance_metrics.py`

The performance metrics module enables quantitative comparison:

- **FPS Calculation** - Rolling average for stability
- **Inference Latency** - Per-frame timing in milliseconds
- **Threshold Tuning** - Confidence threshold optimization (inspired by hyperparameter tuning concepts)
- **Evaluation Metrics** - Qualitative precision comparison based on detection consistency and false positive rates

### 3.4 Lab 12 (Conceptual) — Real-Time ML Pipeline
**Location:** `main.py`, `src/camera/video_stream.py`

The real-time pipeline implements:

```
Camera Capture → Preprocessing → Inference → Visualization → Display
```

This closed-loop system runs continuously at real-time speeds.

---

## 4. Dataset Description

### COCO Pretrained Models
Both models are pretrained on the **COCO (Common Objects in Context)** dataset:

| Attribute | Value |
|-----------|-------|
| Total Classes | 80 |
| Training Images | 118,287 |
| Validation Images | 5,000 |
| Object Categories | Person, Vehicle, Animal, Food, Furniture, Electronics, etc. |

### Target Classes for Demo
For demonstration purposes, the system focuses on common objects:
- person
- bottle
- cup
- cell phone
- laptop
- chair
- keyboard
- mouse
- book
- remote

**Note:** No custom training is required as pretrained COCO weights provide excellent detection for common objects.

---

## 5. Data Preprocessing

### Preprocessing Pipeline

The preprocessing module (`ImageProcessor`) applies the following transformations:

#### 5.1 Image Resizing
```python
def _resize(self, image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
```
- **Purpose:** Ensures consistent input dimensions (640×640)
- **Method:** Bilinear interpolation for quality

#### 5.2 Normalization
```python
def _normalize(self, image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0
```
- **Purpose:** Scales pixel values to [0, 1] range
- **Benefit:** Improved neural network convergence and stability

#### 5.3 Color Space Conversion
```python
def _bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```
- **Purpose:** OpenCV uses BGR; models expect RGB
- **Benefit:** Correct color interpretation during inference

---

## 6. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                                               │
│  │ Mobile Phone │                                               │
│  │ (IP Webcam)  │                                               │
│  └──────┬───────┘                                               │
│         │ Wi-Fi Video Stream                                    │
│         ▼                                                       │
│  ┌──────────────┐     ┌───────────────────────────┐             │
│  │ VideoStream  │────▶│ Image Preprocessing       │             │
│  │ Module       │     │ • Resize (640×640)        │             │
│  └──────────────┘     │ • Normalize               │             │
│                       └───────────────────────────┘             │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────┐            │
│  │         Multi-Model Detection Engine            │            │
│  ├────────────────────┬────────────────────────────┤            │
│  │   YOLOv8-nano      │      YOLOv5-small          │            │
│  └────────────────────┴────────────────────────────┘            │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐     ┌───────────────────────────┐             │
│  │ Visualization│────▶│ Performance Metrics       │             │
│  │ Module       │     │ • FPS Counter             │             │
│  │              │     │ • Latency Display         │             │
│  └──────────────┘     └───────────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Module Descriptions

| Module | File | Responsibility |
|--------|------|----------------|
| VideoStream | `src/camera/video_stream.py` | Camera connection, frame capture |
| ImageProcessor | `src/preprocessing/image_processor.py` | Preprocessing pipeline |
| YOLOv8Detector | `src/detection/yolo_v8_detector.py` | YOLOv8-nano inference |
| YOLOv5Detector | `src/detection/yolo_v5_detector.py` | YOLOv5-small inference |
| Visualizer | `src/utils/visualization.py` | Bounding box drawing |
| PerformanceMetrics | `src/utils/performance_metrics.py` | FPS/latency tracking |

---

## 7. Model Design

### 7.1 YOLOv8-nano

| Attribute | Value |
|-----------|-------|
| Framework | Ultralytics |
| Architecture | CSPDarknet backbone + PANet neck |
| Parameters | ~3.2M |
| Input Size | 640×640 |
| Output | Bounding boxes, class IDs, confidence scores |

**Key Features:**
- Anchor-free detection
- Improved accuracy-to-speed ratio
- Native support for various tasks (detect, segment, pose)

### 7.2 YOLOv5-small

| Attribute | Value |
|-----------|-------|
| Framework | PyTorch Hub |
| Architecture | CSPNet backbone |
| Parameters | ~7.2M |
| Input Size | 640×640 |
| Output | Bounding boxes, class IDs, confidence scores |

**Key Features:**
- Well-established and thoroughly tested
- Strong community support
- Easy deployment options

### 7.3 Common Interface (BaseDetector)

Both models implement a common abstract interface:

```python
class BaseDetector(ABC):
    @abstractmethod
    def load_model(self) -> bool: ...
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]: ...
```

This design enables runtime model switching and fair comparison.

---

## 8. Model Evaluation

### Evaluation Methodology

Since we use pretrained COCO models, our evaluation focuses on:
1. **Inference Speed** - FPS and latency measurements
2. **Detection Consistency** - Stability of detections across frames
3. **Confidence Distribution** - Quality of confidence predictions
4. **False Positive Rate** - Incorrect detections

### Expected Performance Metrics

| Metric | YOLOv8-nano | YOLOv5-small |
|--------|-------------|--------------|
| Inference Time | 15-25 ms | 20-35 ms |
| FPS (GPU) | 40-60 | 28-50 |
| COCO mAP | 37.3 | 37.4 |
| Model Size | 6.3 MB | 14.1 MB |

### Threshold Tuning

Confidence threshold affects detection quality:

| Threshold | Effect |
|-----------|--------|
| 0.25 | More detections, more false positives |
| 0.50 | Balanced (default) |
| 0.75 | Fewer detections, higher precision |

---

## 9. Real-Time Performance Analysis

### Performance Metrics Tracked

The system tracks these metrics in real-time:

1. **FPS (Frames Per Second)**
   - Calculated using rolling average (30-frame window)
   - Displayed on-screen for continuous monitoring

2. **Inference Latency**
   - Time from frame input to detection output
   - Measured in milliseconds

3. **Detection Count**
   - Number of objects detected per frame
   - Useful for scene complexity analysis

### Expected Results

| Environment | YOLOv8-nano FPS | YOLOv5-small FPS |
|-------------|-----------------|-------------------|
| Cloud GPU (T4) | 45-60 | 35-50 |
| Local GPU | 30-45 | 25-40 |
| CPU Only | 5-10 | 3-8 |

---

## 10. Execution Environment

> **Important:** Due to limited local CPU resources and the real-time nature of object detection models, the system was deployed and evaluated on a **cloud-based GPU environment (Google Colab)**. This ensured stable real-time performance and allowed fair comparison between multiple detection models. The use of cloud acceleration reflects real-world ML deployment practices.

### Recommended Setup

| Scenario | Recommendation |
|----------|----------------|
| Weak CPU | ✅ Use Cloud GPU |
| Short on time | ✅ Use Cloud GPU |
| Want smooth demo | ✅ Use Cloud GPU |
| Worried about lag | ✅ Use Cloud GPU |

### Hardware Requirements

- **For Cloud:** Google Colab with GPU runtime (Tesla T4)
- **For Local:** NVIDIA GPU with CUDA support (optional)
- **Camera:** Android phone with IP Webcam app
- **Network:** Same Wi-Fi for phone and PC

### Software Requirements

- Python 3.8+
- PyTorch 2.0+
- Ultralytics (YOLOv8)
- OpenCV 4.8+
- NumPy, Matplotlib

---

## 11. Challenges & Limitations

### Technical Challenges

| Challenge | Solution |
|-----------|----------|
| Network latency | Use local Wi-Fi, minimize video resolution |
| GPU memory | Use lightweight models (nano, small variants) |
| Frame drops | Implement frame skipping if needed |
| Connection stability | Retry logic in VideoStream class |

### Known Limitations

1. **Network Dependency** - Requires stable Wi-Fi connection between phone and PC
2. **Lighting Conditions** - Detection accuracy varies with lighting
3. **Small Objects** - Nano/small models may miss very small objects
4. **Occlusion** - Partially visible objects may not be detected

### Mitigation Strategies

- Use 720p or lower resolution for stable streaming
- Ensure good lighting for demo
- Position camera at appropriate distance
- Allow 2-3 seconds for model warm-up

---

## 12. Ethical Considerations

### Privacy Concerns

- **Camera Usage:** Only process frames for object detection, no storage
- **Personal Data:** No facial recognition or person identification
- **Data Handling:** Frames are processed and discarded, not saved

### Responsible AI Principles

1. **Transparency** - Model predictions are displayed with confidence scores
2. **Fairness** - Using well-validated pretrained models
3. **Accountability** - Clear documentation of system limitations
4. **Safety** - Not used for critical decision-making without human oversight

### Potential Misuse Prevention

- System designed for educational purposes only
- No surveillance or tracking capabilities
- Detection limited to general object categories

---

## 13. Conclusion & Future Work

### Summary

This project successfully demonstrates:

1. **Real-time object detection** from mobile camera input
2. **Multi-model implementation** with YOLOv8-nano and YOLOv5-small
3. **Performance comparison** between different model architectures
4. **Integration of 4 lab concepts** in a cohesive system

### Key Achievements

- ✅ Complete implementation of real-time detection pipeline
- ✅ Professional code structure with modular design
- ✅ Performance metrics tracking and comparison
- ✅ User-friendly interface with runtime model switching

### Future Improvements

1. **Custom Training** - Fine-tune models on domain-specific objects
2. **Model Optimization** - Apply quantization for faster inference
3. **Multi-Camera Support** - Handle multiple video streams
4. **Object Tracking** - Add tracking across frames
5. **Web Interface** - Build browser-based UI with Flask/Streamlit
6. **Edge Deployment** - Deploy on mobile devices or Raspberry Pi

---

## 14. References

1. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv:1804.02767
2. Jocher, G. (2022). YOLOv5 by Ultralytics. GitHub repository.
3. Jocher, G. (2023). YOLOv8 by Ultralytics. GitHub repository.
4. Lin, T. Y., et al. (2014). Microsoft COCO: Common Objects in Context. ECCV.
5. OpenCV Documentation. https://docs.opencv.org/
6. PyTorch Documentation. https://pytorch.org/docs/
7. ML Lab Manual, BSAI-462, Dawood University of Engineering & Technology

---

## Appendix A: Setup Instructions

### Installation

```bash
# Clone/download the project
cd Real-time-Multi-Model-Object-Classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# With webcam
python main.py --model yolo8 --source 0

# With IP Webcam
python main.py --model yolo5 --source "http://192.168.1.10:8080/video"

# Compare both models
python main.py --model both --source 0
```

### Mobile Camera Setup (IP Webcam)

1. Install "IP Webcam" app from Google Play Store
2. Open the app and tap "Start Server"
3. Note the IP address shown (e.g., http://192.168.1.10:8080)
4. Use `{IP}:8080/video` as the source URL

---

## Appendix B: Project Structure

```
Real-time Multi-Model Object Classifier/
├── main.py                     # Main entry point
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
│
├── src/
│   ├── camera/
│   │   └── video_stream.py     # Camera handling
│   ├── preprocessing/
│   │   └── image_processor.py  # Image preprocessing
│   ├── detection/
│   │   ├── base_detector.py    # Abstract detector
│   │   ├── yolo_v8_detector.py # YOLOv8 implementation
│   │   └── yolo_v5_detector.py # YOLOv5 implementation
│   └── utils/
│       ├── config.py           # Configuration
│       ├── performance_metrics.py
│       └── visualization.py
│
├── report/
│   └── PROJECT_REPORT.md       # This report
│
├── models/                     # Auto-downloaded weights
├── results/                    # Performance metrics
└── demo/                       # Demo videos
```

---

*End of Report*
