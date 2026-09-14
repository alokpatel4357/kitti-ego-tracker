# Project Statement

## Problem Statement
In autonomous driving and mobile robotics, dynamic obstacle detection is severely hindered by camera ego-motion. Traditional background subtraction fails because camera translation and rotation cause all static background pixels to move across the image plane. There is a need for a lightweight, mathematically verifiable system that decouples camera-induced motion from independent object motion without relying on computationally expensive, black-box neural networks.

## Target Users
* **Autonomous Vehicle Engineers:** Requiring non-neural, geometrically verifiable fallback systems for obstacle detection.
* **Robotics Researchers:** Needing lightweight visual tracking pipelines capable of running on resource-constrained edge hardware.
* **Computer Vision Practitioners:** Seeking practical algorithmic implementations of epipolar geometry and subspace motion decomposition.

## Scope of the Project
The project processes sequential monocular automotive image sequences (such as the KITTI dataset) and outputs tracked video files with spatial bounding telemetry. The scope focuses strictly on dense flow field decomposition, epipolar RANSAC outlier elimination, and unsupervised spatial clustering. It excludes stereo disparity mapping, multi-sensor fusion (LiDAR/Radar), and deep-learning semantic classification.

## High-Level Features
* **Unsupervised Ego-Motion Modeling:** Reconstructs camera-induced velocity fields via Principal Component Analysis.
* **Epipolar Geometry Verification:** Uses RANSAC Fundamental Matrix estimation to isolate rigid 3D scene points.
* **Residual Anomaly Isolation:** Extracts independent object movements by subtracting projected background motion.
* **Headless Video Export:** Operates strictly via CLI without graphical display servers, exporting annotated MP4 streams directly to disk.
