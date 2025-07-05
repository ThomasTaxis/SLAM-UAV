# MSCKF-SLAM for UAVs

This project implements a lightweight **MSCKF-SLAM** (Multi-State Constraint Kalman Filter) system designed for Unmanned Aerial Vehicles (UAVs), with a focus on minimal dependencies, and customizability.

---

## 📌 Overview

MSCKF-SLAM is a tightly coupled visual-inertial SLAM algorithm that fuses IMU and camera data for real-time localization and mapping. This repository contains a full Python implementation for educational and prototyping purposes, structured around a central script `MSCKF-SLAM.py`.


---

## 🚁 Key Features

- Full MSCKF implementation in Python
- Visual-inertial sensor fusion (monocular camera + IMU)
- Sliding window state estimation


---
## Instructions

1. **Download the Dataset**

   python download_data.py

2. **Option A: Run `main.py` (Feature Extraction protocol - Full Pipeline – Time-Consuming)**

   This will execute the full processing pipeline and generate the `visual_corrections.json` file: python main.py

   Then proceed to run python main2.py to retrieve tha UAV trajectory after the MSKFC application

   **Option B: Skip running main.py and use the precomputed visual_corrections.json file included in the repository. Then simply run: python main2.py

