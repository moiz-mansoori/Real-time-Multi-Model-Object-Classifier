# Real-time Multi-Model Object Classifier (Mobile-Based)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Lab-14: Complex Computing Activity**  
> Machine Learning (BSAI-462) | 7th Semester | Batch 2022
>
> **Group Members:**  
> Hamza Kamelen (22F-BSAI-09) | Moiz Ahmed Mansoori (22F-BSAI-32)  
> Muzzamil Khalid (22F-BSAI-29) | Muhammad Sami (22F-BSAI-43)

A real-time object detection system using mobile phone camera as video input, implementing multiple ML models with performance comparison.

---

## 📌 Project Description

This project implements a **real-time object detection system** using a **mobile phone camera** as the video input source. It uses **multiple machine learning models** to detect objects in live video streams and compares their performance.

🚫 No Augmented Reality (AR) is used  
📱 Mobile phone is used only as a camera  
⚡ Focus is on real-time ML deployment  
☁️ Optimized for cloud GPU execution (Google Colab)

---

## 🎯 Features

- ✅ Real-time object detection
- ✅ Mobile phone camera integration (IP Webcam)
- ✅ **Multiple ML models**:
    - YOLOv8-nano (General Purpose)
    - YOLOv5-small (General Purpose)
    - **Gun Detector** (Security Focused)
- ✅ Bounding boxes with labels and confidence scores
- ✅ FPS and latency measurement
- ✅ Runtime model switching
- ✅ Performance comparison

---

## 📁 Project Structure

```
Real-time Multi-Model Object Classifier/
│
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── src/
│   ├── camera/
│   │   └── video_stream.py          # IP Webcam connection
│   │
│   ├── preprocessing/
│   │   └── image_processor.py       # Image preprocessing (Lab 1)
│   │
│   ├── detection/
│   │   ├── base_detector.py         # Abstract detector class
│   │   ├── yolo_v8_detector.py      # YOLOv8-nano (Lab 5)
│   │   ├── yolo_v5_detector.py      # YOLOv5-small (Lab 5)
│   │   └── gun_detector.py          # Gun Detector (Security)
│   │
│   └── utils/
│       ├── config.py                # Configuration
│       ├── performance_metrics.py   # FPS & latency (Lab 6)
│       └── visualization.py         # Bounding box drawing
│
├── report/
│   └── PROJECT_REPORT.md            # Lab-14 report (3-5 pages)
│
├── models/                          # Auto-downloaded weights
├── results/                         # Performance metrics
└── demo/                            # Demo videos
```

---

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/moiz-mansoori/Real-time-Multi-Model-Object-Classifier.git
cd Real-time-Multi-Model-Object-Classifier
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📱 Mobile Camera Setup (IP Webcam)

1. Install **IP Webcam** app from Google Play Store
2. Open the app and tap **"Start Server"**
3. Note the IP address shown (e.g., `http://192.168.1.10:8080`)
4. Ensure phone and laptop are on the **same Wi-Fi network**

---

## ▶️ How to Run

### Basic Usage

```bash
# Run with YOLOv8 and webcam
python main.py --model yolo8 --source 0

# Run with YOLOv5 and webcam
python main.py --model yolo5 --source 0

# Run Gun Detector
python main.py --model gun --source 0

# Run with IP Webcam
python main.py --model yolo8 --source "http://192.168.1.10:8080/video"

# Compare all models
python main.py --model both --source 0
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `1` | Switch to YOLOv8 |
| `2` | Switch to YOLOv5 |
| `3` | Switch to Gun Detector |
| `+` | Increase confidence threshold |
| `-` | Decrease confidence threshold |
| `Q` | Quit |

---

## Performance Metrics

Expected performance (hardware dependent):

| Metric | YOLOv8-nano | YOLOv5-small | Gun Detector |
|--------|-------------|--------------|--------------|
| FPS (Cloud GPU) | 45-60 | 35-50 | 45-60 |
| FPS (Local GPU) | 30-45 | 25-40 | 30-45 |
| FPS (CPU) | 5-10 | 3-8 | 5-10 |

---

## Project Report

See [`report/PROJECT_REPORT.md`](report/PROJECT_REPORT.md) for the complete Lab-14 report including:

- Introduction & Problem Statement
- Lab Concepts Integration
- System Architecture (Detailed Diagrams)
- Model Design & Evaluation
- Performance Analysis
- Challenges & Limitations
- Ethical Considerations
- Conclusion & Future Work

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## Acknowledgments

- Ultralytics for YOLOv8
- PyTorch team for YOLOv5
- OpenCV community
- Course Instructor: Engr. Hamza Farooqui
