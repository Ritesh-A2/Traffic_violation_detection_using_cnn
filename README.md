# VisionSpeed: Automated Real-Time Vehicle Speed Estimation & Traffic Analysis

An end-to-end computer vision pipeline designed for real-time traffic surveillance, bidirectional vehicle counting, and speed estimation using YOLOv8 object detection and spatial centroid tracking.

---

## Overview

Accurate velocity estimation in roadside camera feeds is a fundamental requirement for intelligent transportation systems (ITS). **VisionSpeed** eliminates dependencies on expensive hardware sensors (like radar or LiDAR) by leveraging monocular video geometry and deep-learning-based object localization.

The system establishes two virtual spatial calibration markers across the road corridor. By computing the exact frame elapsed time as a vehicle crosses both boundaries along its motion vector, the application derives real-world speed (km/h) and flags traffic violations dynamically.

---

## Key Features

- **Dual Detection Support:** Supports both fine-tuned custom YOLOv8 models (`best.pt`) and pre-trained official weights (`yolov8n.pt` with COCO class filtering).
- **Persistent Centroid Tracking:** A lightweight spatial centroid tracking engine that preserves vehicle class taxonomy, assigns unique persistent IDs, and prevents identity swaps across frames.
- **Bidirectional Velocity Estimation:** Automatically handles both **Downward** (Red to Blue) and **Upward** (Blue to Red) traffic streams.
- **Robust Line-Crossing Triggers:** Replaces narrow proximity thresholds with a directional vector cross-over algorithm (`prev_y` vs `curr_y`), preventing tracking misses caused by fast-moving vehicles.
- **Clean Visual HUD (No Cluttered Bounding Boxes):** Renders clean ID, classification, and speed telemetry directly above target vehicles.
- **Permanent Violation Alert:** Dynamically switches label colors from **Green** to **Red** upon detecting over-speeding (> 40 km/h) and maintains the violation state until the vehicle exits the field of view.
- **Interactive Source Selection:** Native OS file explorer integration via Tkinter for rapid testing with local traffic video files.

---

## System Architecture & Mathematical Formulation

### 1. Spatial Tracking
For each detected bounding box $[x_1, y_1, x_2, y_2]$, the vehicle reference point is computed as its spatial centroid:

$$c_x = \frac{x_1 + x_2}{2}, \quad c_y = \frac{y_1 + y_2}{2}$$

Target associations between frame $t-1$ and $t$ are evaluated using Euclidean distance:

$$D = \sqrt{(c_{x, t} - c_{x, t-1})^2 + (c_{y, t} - c_{y, t-1})^2}$$

### 2. Temporal Reference
To ensure frame-rate invariance regardless of hardware processing lags, temporal points are derived from the hardware stream index rather than system clock time:

$$T_{\text{frame}} = \frac{\text{Frame Index}}{\text{Video FPS}}$$

### 3. Velocity Equation
Given calibrated ground distance $D_{\text{real}}$ (default: 10 meters) between the two spatial reference markers:

$$\Delta T = \vert{}T_{\text{line\_2}} - T_{\text{line\_1}}\vert{}$$

$$\text{Speed } (\text{km/h}) = \left( \frac{D_{\text{real}}}{\Delta T} \right) \times 3.6$$

---

## Project Structure

```text
├── Results/
│   ├── best.pt              # Custom trained YOLOv8 weights (Optional)
│   └── yolov8n.pt           # Official pre-trained YOLOv8 nano model
├── videos/             # Sample traffic surveillance footage (.mp4, .avi)
├── predict.py                  # Core inference and tracking pipeline
├── train(1).py                 # Training script for custom dataset (if applicable)
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
