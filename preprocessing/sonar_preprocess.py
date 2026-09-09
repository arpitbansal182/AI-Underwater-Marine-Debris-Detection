import cv2
import shutil
from pathlib import Path


# Noise reduction + contrast enhancement
def preprocess_sonar(input_path, output_path):
    image = cv2.imread(str(input_path))

    if image is None:
        print(f"WARNING: Could not read {input_path}")
        return False

    # Step 1: Median filtering to reduce sonar noise
    denoised = cv2.medianBlur(image, 5)

    # Step 2: CLAHE for local contrast enhancement
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), result)

    return True


def process_dataset():
    # Original dataset
    source_root = Path("data/processed")

    # New noise-reduced dataset
    output_root = Path("data/cleaned")

    splits = ["train", "valid", "test"]

    total_images = 0

    for split in splits:

        source_images = source_root / split / "images"
        source_labels = source_root / split / "labels"

        output_images = output_root / split / "images"
        output_labels = output_root / split / "labels"

        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {split}...")

        image_files = list(source_images.glob("*.jpg"))

        processed = 0

        for image_path in image_files:

            output_image = output_images / image_path.name

            success = preprocess_sonar(
                image_path,
                output_image
            )

            if success:
                processed += 1

                # Copy corresponding YOLO label
                label_path = source_labels / f"{image_path.stem}.txt"

                if label_path.exists():
                    shutil.copy2(
                        label_path,
                        output_labels / label_path.name
                    )

        total_images += processed

        print(f"{split}: {processed} images processed")

    print("\n================================")
    print("CLEAN DATASET CREATED")
    print("================================")
    print(f"Total images: {total_images}")
    print(f"Location: {output_root}")


if __name__ == "__main__":
    process_dataset()