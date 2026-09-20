# WA-coding-challenge
# Autonomous Driving BEV Trajectory & Object Tracking Pipeline

This project implements a Bird's-Eye-View (BEV) vehicle localization and multi-object trajectory tracking pipeline using 2D/3D camera depth fusion and deep learning. 

The project is divided into two distinct parts: **Part A (Ego-Vehicle Trajectory Extraction)** and **Part B (YOLOv8 Dynamic Target Tracking in BEV)**.

---

## 🧩 Part A: Ego-Vehicle Trajectory Extraction (2D/3D Camera Fusion)

### 1. What I Did
* **Traffic Light Detection Parsing**: Loaded 2D bounding boxes of traffic lights from `bbox_light.csv` (used as a static ground origin $(0, 0)$ reference point).
* **3D Point Cloud Integration**: Synchronized depth map files (`.npz`) frame-by-frame with bounding box coordinates.
* **Noise Reduction via Local Patching**: Calculated the center pixel $(u, v)$ of the traffic light and extracted a local $3 \times 3$ pixel patch from the 3D point cloud array. Filtered out invalid `NaN`s and zero-depth readings to reliably compute the mean camera coordinate $[X_c, Y_c, Z_c]$.
* **Ground Coordinate Transformation**: Inverted the camera-relative coordinates $(-X_c, -Y_c)$ to compute the ego-vehicle's absolute position in a BEV (Bird's-Eye-View) ground frame.
* **Visualization Output**:
  * Generated a high-resolution static trajectory plot saved as `trajectory.png`.
  * Built an animated trajectory rendering loop using Matplotlib and OpenCV to export `trajectory.mp4` at 10 FPS.

### 2. Key Assumptions
* **Static Reference Point**: The traffic light remains static across all processed frames and serves as a reliable origin marker $(0, 0)$.
* **Valid Depth Overlap**: The localized $3 \times 3$ depth patch contains enough valid 3D point cloud measurements to reflect accurate physical distances.

---

## 🧩 Part B: YOLOv8 Object Tracking & BEV Trajectory Mapping

### 1. What I Did
* **YOLOv8 Object Detection Integration**: Integrated the `yolov8n` deep learning detection model to perform automatic vehicle localization on RGB stereo images (`left000001.png`).
* **Target Filtering (Golf Cart/Vehicle Selection)**: Filtered COCO object classes (cars, buses, trucks) and selected the primary vehicle candidate based on bounding box area.
* **Relative 3D Localization**: Extracted $3 \times 3$ depth patches at the detected vehicle's center pixel to compute its camera-frame 3D coordinates $(X_c, Y_c)$.
* **Global Trajectory Composition**: Combined the target's relative offset with the Ego-vehicle's absolute ground position calculated in Part A:
  $$\text{Target}_{\text{world}} = \text{Ego}_{\text{world}} + \text{Target}_{\text{camera}}$$
* **Robust Fallback Handling**: Implemented position persistence strategies to smooth out trajectories during transient occlusions or missed YOLO detections.
* **Dual-Trajectory Visualization**:
  * Generated a static BEV map comparing Ego and Golf Cart paths (`bev_rich.png`).
  * Created a dynamic multi-agent trajectory animation exported as `bev_rich.mp4`.

### 2. Key Assumptions
* **Bounding Box Depth Validity**: The center region of the detected vehicle's bounding box maps to valid physical depth surfaces.
* **Temporal Continuity**: In frames where detection fails or depth values are missing, target position can be inferred from prior frame estimates.

---

## 📊 Outputs & Artifacts

| Part | Output File | Description |
| :--- | :--- | :--- |
| **Part A** | `trajectory.png` | Static BEV trajectory plot of the Ego-vehicle |
| **Part A** | `trajectory.mp4` | Animated BEV trajectory video of the Ego-vehicle |
| **Part B** | `bev_rich.png` | Static dual-trajectory BEV map (Ego-vehicle vs. Golf Cart) |
| **Part B** | `bev_rich.mp4` | Animated dynamic BEV tracking video of both vehicles |

---

## 🚀 Quick Start

### 1. Requirements
Ensure required packages are installed:
```bash
pip install numpy pandas matplotlib opencv-python ultralytics
