from pathlib import Path
from PIL import Image, ImageDraw


# Paths
image_path = Path(
    "data/processed/train/images/"
    "baycove_01_18_png_jpg.rf.68c525012668cd138c2a5fca781abd24.jpg"
)

label_path = Path(
    "data/processed/train/labels/"
    "baycove_01_18_png_jpg.rf.68c525012668cd138c2a5fca781abd24.txt"
)


# Open image
image = Image.open(image_path)
draw = ImageDraw.Draw(image)

image_width, image_height = image.size

print("Image size:", image_width, "x", image_height)


# Read YOLO labels
with open(label_path, "r") as file:

    for line in file:

        class_id, x_center, y_center, width, height = map(
            float, line.split()
        )

        # Convert normalized coordinates to pixels
        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        # Calculate corners
        x1 = x_center - width / 2
        y1 = y_center - height / 2

        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Draw bounding box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        print("Class:", int(class_id))
        print("Bounding box:", x1, y1, x2, y2)


# Save visualization
output_path = Path(
    "data/processed/train/"
    "annotation_check.jpg"
)

image.save(output_path)

print("Saved:", output_path)