# AI-Powered Underwater Marine Debris and Anomaly Detection

An AI-based prototype for detecting marine debris and derelict fishing gear from **Side-Scan Sonar (SSS) imagery** using **YOLOv9t**.

The system performs sonar image processing, object detection, confidence and bounding-box extraction, and stores detection results in a SQLite database. A Streamlit-based interface is provided for testing the complete detection pipeline.

---

## Project Overview

Marine debris and abandoned fishing gear can pose significant risks to marine ecosystems and underwater operations. Manually inspecting large volumes of underwater sonar imagery can be time-consuming.

This project explores an automated computer-vision approach using Side-Scan Sonar imagery.

The prototype focuses on detecting:

- `Crab-Pot`
- `Maybe-Crab-Pot`

The current model is trained using the GhostVision Side-Scan Sonar Crab-Pot Detection Dataset.

> **Note:** The current prototype should not be interpreted as a universal marine-debris detector. Its trained classes are limited to the classes available in the training dataset.

---

## System Pipeline

```text
Side-Scan Sonar Image
          |
          v
Image Preprocessing
(Median Filter + CLAHE)
          |
          v
YOLOv9t Object Detection
          |
          v
Detected Object
(Class + Confidence + Bounding Box)
          |
          v
SQLite Database
          |
          v
Backend / API Integration
          |
          v
Frontend / Visualization
