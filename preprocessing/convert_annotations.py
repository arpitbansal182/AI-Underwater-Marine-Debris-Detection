import json
import shutil
from pathlib import Path

from PIL import Image


# -----------------------------
# Project paths
# -----------------------------

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

SPLITS = ["train", "valid", "test"]

# YOLO class IDs
CLASS_MAP = {
    "Crab-Pot": 0,
    "Maybe-Crab-Pot": 1
}


def convert_bbox_to_yolo(bbox, image_width, image_height):
    """
    Convert:
        [x, y, width, height]

    into YOLO format:
        [x_center, y_center, width, height]

    All YOLO coordinates are normalized to 0-1.
    """

    x, y, width, height = bbox

    x_center = x + width / 2
    y_center = y + height / 2

    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return x_center, y_center, width, height


def process_split(split):
    raw_split_dir = RAW_DIR / split

    image_output_dir = OUTPUT_DIR / split / "images"
    label_output_dir = OUTPUT_DIR / split / "labels"

    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = raw_split_dir / "metadata.jsonl"

    print(f"\nProcessing: {split}")

    image_count = 0
    object_count = 0

    with open(metadata_file, "r", encoding="utf-8") as file:

        for line in file:
            record = json.loads(line)

            filename = record["file_name"]
            objects = record["objects"]

            image_path = raw_split_dir / filename

            if not image_path.exists():
                print(f"WARNING: Image not found: {filename}")
                continue

            # Read image dimensions
            with Image.open(image_path) as image:
                image_width, image_height = image.size

            # Copy image to YOLO dataset folder
            output_image_path = image_output_dir / filename
            shutil.copy2(image_path, output_image_path)

            # YOLO label file has same filename but .txt
            label_filename = Path(filename).stem + ".txt"
            label_path = label_output_dir / label_filename

            with open(label_path, "w", encoding="utf-8") as label_file:

                for bbox, category in zip(
                    objects["bbox"],
                    objects["category"]
                ):

                    if category not in CLASS_MAP:
                        print(f"WARNING: Unknown class: {category}")
                        continue

                    class_id = CLASS_MAP[category]

                    x_center, y_center, width, height = (
                        convert_bbox_to_yolo(
                            bbox,
                            image_width,
                            image_height
                        )
                    )

                    label_file.write(
                        f"{class_id} "
                        f"{x_center:.6f} "
                        f"{y_center:.6f} "
                        f"{width:.6f} "
                        f"{height:.6f}\n"
                    )

                    object_count += 1

            image_count += 1

    print(f"Images processed: {image_count}")
    print(f"Objects processed: {object_count}")


def main():
    for split in SPLITS:
        process_split(split)

    print("\nConversion complete!")


if __name__ == "__main__":
    main()