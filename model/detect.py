from ultralytics import YOLO
from pathlib import Path


# Path to our trained YOLOv9t model
MODEL_PATH = Path(
    "runs/detect/runs/sih_yolov9t_full/weights/best.pt"
)


def detect_sonar(image_path, confidence=0.15):

    # Load trained YOLOv9t model
    model = YOLO(str(MODEL_PATH))

    # Run detection
    results = model.predict(
        source=image_path,
        conf=confidence,
        device=0,
        verbose=False
    )

    result = results[0]

    detections = []

    # Extract detected objects
    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        confidence_score = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class": class_name,
            "confidence": round(confidence_score, 4),
            "bbox": [
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2)
            ]
        })

    return {
        "image": str(image_path),
        "detections": detections
    }


# Test the detection module
if __name__ == "__main__":

    test_image = (
        "data/processed/train/images/"
        "baycove_17_12_png_jpg.rf.1f0d3002109e3004926f60df41715b09.jpg"
    )

    result = detect_sonar(
        test_image,
        confidence=0.15
    )

    print(result)