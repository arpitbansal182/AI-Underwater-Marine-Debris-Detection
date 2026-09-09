# AI-Powered Underwater Marine Debris Detection

## Project
AI-based detection of marine debris / derelict fishing gear from side-scan sonar imagery.

## Current Prototype
The trained prototype uses YOLOv9t.

Current trained classes:
- Crab-Pot
- Maybe-Crab-Pot

## AI Pipeline

Side-Scan Sonar Image
        ?
Median Filter + CLAHE
        ?
YOLOv9t
        ?
Object Detection
        ?
Class + Confidence + Bounding Box
        ?
SQLite Database
        ?
Backend/API Integration

## Trained Model

Model:
YOLOv9t

Weights:
runs/detect/runs/detect/sih_yolov9t_clean/weights/best.pt

Validation performance:
mAP@50: 46.64%
Precision: 60.73%
Recall: 47.61%
mAP@50-95: 16.51%

## Database

Database:
detections.db

Table:
detections

Columns:
- id
- image_name
- object_class
- confidence
- x1
- y1
- x2
- y2
- latitude
- longitude
- timestamp

## Running the Prototype

Activate the virtual environment and install dependencies:

pip install -r requirements.txt

Run:

streamlit run app.py

## Backend Integration

The backend can use model/detect.py for inference.

The trained model should remain at:

runs/detect/runs/detect/sih_yolov9t_clean/weights/best.pt

The backend should pass sonar images to the detection module and use the returned:
- class
- confidence
- bounding box

The existing SQLite database can be retained during prototype integration.

## Important

This prototype currently demonstrates detection of crab pots / derelict fishing gear using the available annotated side-scan sonar dataset. It should not be described as detecting every possible type of marine debris.
