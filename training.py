import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Add this line at the very top

from ultralytics import YOLO
import os


def train_yolov11(data_yaml_path, model_size='n', epochs=50, batch_size=16):
    """Train YOLOv11 on KITTI dataset"""

    model = YOLO(f'yolo11{model_size}.pt')

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        name='kitti_yolo11',
        project='runs/detect',
        pretrained=True,
        optimizer='AdamW',
        lr0=0.01,
        patience=50,
        save=True,
        save_period=10
    )

    return results


if __name__ == "__main__":
    # Set working directory
    os.chdir('E:/workspace/Yolo4Euro')

    # Train the model
    data_yaml = "yolo_dataset/data.yaml"
    results = train_yolov11(
        data_yaml_path=data_yaml,
        model_size='n',
        epochs=50,
        batch_size=16
    )

    print("Training complete!")
    print(f"Model saved to: runs/detect/kitti_yolo11/weights/best.pt")