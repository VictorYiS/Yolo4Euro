import os
import cv2
import numpy as np
import shutil
from pathlib import Path


def convert_kitti_to_yolo(kitti_data_dir, output_dir):
    """
    Convert KITTI dataset format to YOLOv11 format

    KITTI label format:
    class truncated occluded alpha bbox_2D(4) dimensions_3D(3) location_3D(3) rotation_y

    YOLO label format:
    class x_center y_center width height (normalized)
    """

    # Create output directory structure
    output_dir = Path(output_dir)
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    # KITTI class mapping for cars only (simplest case)
    class_mapping = {
        'Car': 0,
        'Van': 1,
        'Truck': 2,
        'Pedestrian': 3,
        'Person_sitting': 4,
        'Cyclist': 5,
        'Tram': 6,
        'Misc': 7,
        'DontCare': 8
    }

    def convert_bbox(size, box):
        """Convert KITTI bbox to YOLO format (normalized)"""
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        x = (box[0] + box[2]) / 2.0
        y = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    # Process training data
    kitti_train_img_dir = Path(kitti_data_dir) / "training" / "image_2"
    kitti_train_label_dir = Path(kitti_data_dir) / "training" / "label_2"

    for img_file in kitti_train_img_dir.glob("*.png"):
        img_name = img_file.stem
        label_file = kitti_train_label_dir / f"{img_name}.txt"

        # Read image to get dimensions
        img = cv2.imread(str(img_file))
        h, w, _ = img.shape

        # Copy image
        shutil.copy(str(img_file), output_dir / "images" / "train" / img_file.name)

        # Convert labels
        yolo_labels = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 15:
                        continue

                    class_name = parts[0]
                    if class_name in class_mapping:
                        class_id = class_mapping[class_name]

                        # Extract bbox (x1, y1, x2, y2)
                        bbox = [float(parts[4]), float(parts[5]),
                                float(parts[6]), float(parts[7])]

                        # Convert to YOLO format
                        yolo_bbox = convert_bbox((w, h), bbox)
                        yolo_labels.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")

        # Write YOLO label file
        with open(output_dir / "labels" / "train" / f"{img_name}.txt", 'w') as f:
            f.write('\n'.join(yolo_labels))

    # Process test data (copy images only, no labels for test set)
    kitti_test_img_dir = Path(kitti_data_dir) / "testing" / "image_2"

    for img_file in kitti_test_img_dir.glob("*.png"):
        shutil.copy(str(img_file), output_dir / "images" / "test" / img_file.name)

    # Create data.yaml file for YOLOv11
    data_yaml_content = f"""
    train: {str(output_dir)}/images/train
    test: {str(output_dir)}/images/test

    nc: {len(class_mapping)}  # number of classes
    names: {list(class_mapping.keys())}  # class names
    """

    with open(output_dir / "data.yaml", 'w') as f:
        f.write(data_yaml_content)

    print(f"Conversion complete! All {len(class_mapping)} classes converted.")

    print(f"Conversion complete! YOLOv11 dataset saved to: {output_dir}")
    print(f"Training images: {len(list((output_dir / 'images' / 'train').glob('*.png')))}")
    print(f"Test images: {len(list((output_dir / 'images' / 'test').glob('*.png')))}")


# Usage example
if __name__ == "__main__":
    kitti_data_dir = "data_object"  # Your KITTI data directory
    output_dir = "yolo_dataset"  # Output directory for YOLOv11 format

    convert_kitti_to_yolo(kitti_data_dir, output_dir)