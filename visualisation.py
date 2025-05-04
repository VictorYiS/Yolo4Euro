import cv2
import numpy as np
from pathlib import Path


def visualize_yolo_labels(image_path, label_path, output_path):
    """Visualize YOLO format labels on image"""

    # Read image
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    # Read labels
    with open(label_path, 'r') as f:
        labels = f.readlines()

    # Draw boxes
    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        # Convert from normalized to absolute coordinates
        x_center *= w
        y_center *= h
        width *= w
        height *= h

        # Calculate box corners
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Class: {class_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save result
    cv2.imwrite(str(output_path), img)


# Test visualization
image_path = "yolo_dataset/images/val/000010.png"
label_path = "yolo_dataset/labels/val/000010.txt"
output_path = "test_visualization.png"

visualize_yolo_labels(image_path, label_path, output_path)