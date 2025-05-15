import os
import time
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from ultralytics import YOLO

from log import log


class LaneDetector:
    """
    Lane detection class that uses a U-Net model to detect lane lines in images.
    """

    def __init__(self, model_path=None, use_gpu=True):
        """
        Initialize the lane detector with a model.

        Args:
            model_path: Path to the lane detection model. If None, tries default paths.
            use_gpu: Whether to use GPU for inference
        """
        # Configure GPU usage
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

        # Load the model with proper error handling
        self.model = self._load_model_safely(model_path)

    def _load_model_safely(self, model_path):
        """
        Safely load the model with proper error handling and IO device configuration.
        """
        # Define possible model paths to try
        if model_path:
            paths_to_try = [model_path]
        else:
            paths_to_try = [
                '../LaneLineLableModels/run_allepoch-5425-val_loss-0.0606-val_acc-0.9794.hdf5',
                'LaneLineLableModels/run_allepoch-5425-val_loss-0.0606-val_acc-0.9794.hdf5'
            ]

        # Add absolute paths if relative paths don't work
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for rel_path in paths_to_try.copy():
            abs_path = os.path.join(current_dir, rel_path)
            if abs_path not in paths_to_try:
                paths_to_try.append(abs_path)

        # Try loading with different options
        for path in paths_to_try:
            if not os.path.exists(path):
                continue

            try:
                # Try with default options
                return load_model(path)
            except Exception as e1:
                print(f"First attempt to load model failed: {e1}")
                try:
                    # Try with IO device options
                    options = tf.saved_model.LoadOptions(
                        experimental_io_device='/job:localhost'
                    )
                    return tf.keras.models.load_model(path, options=options)
                except Exception as e2:
                    print(f"Second attempt with IO device options failed: {e2}")
                    continue

        raise Exception("Failed to load model from any of the paths. Please provide a valid model path.")

    def process_image(self, image_path=None, image_array=None, resize=(400, 400)):
        """
        处理单张图像以检测车道线。

        Args:
            image_path: 图像文件路径
            image_array: 图像的numpy数组(RGB)
            resize: 处理前调整图像大小的元组(宽度,高度)

        Returns:
            包含原始图像、处理后的车道线检测图像和车道线数据的字典
        """
        try:
            if image_path is not None:
                # 检查文件扩展名
                _, ext = os.path.splitext(image_path)
                if not ext.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                    raise ValueError(f"不支持的文件扩展名: {ext}")

                # 从路径加载图像
                image = Image.open(image_path).convert('RGB')
            elif image_array is not None:
                # 使用提供的图像数组
                # 确保图像是RGB格式(OpenCV通常使用BGR)
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    # 如果已经是RGB，直接使用
                    image = Image.fromarray(image_array.astype('uint8'))
                else:
                    raise ValueError("提供的图像数组格式不正确，应为RGB")
            else:
                raise ValueError("必须提供image_path或image_array")

            # 调整图像大小
            resized_image = image.resize(resize)

            # 准备模型输入
            model_input = np.array(resized_image)[None, ...]

            # 安全获取车道线预测
            try:
                lane_predictions = self.model(model_input)
            except Exception as e:
                print(f"模型推理过程中出错: {e}")
                # 如果模型失败则返回空结果
                return {
                    'original_image': image,
                    'resized_image': resized_image,
                    'lane_image': None,
                    'lane_data': None,
                    'error': str(e)
                }

            # 将预测转换为车道线图
            lanes = np.argmax(lane_predictions[0, ...], axis=-1) * 255

            # 从车道线数据创建图像
            lane_image = Image.fromarray(lanes.astype('uint8'), 'L')

            return {
                'original_image': image,
                'resized_image': resized_image,
                'lane_image': lane_image,
                'lane_data': lanes
            }
        except Exception as e:
            print(f"处理图像时出错: {e}")
            return {'error': str(e)}

    def process_folder(self, folder_path, output_folder=None):
        """
        Process all images in a folder.

        Args:
            folder_path: Path to the folder containing images
            output_folder: Path to save processed images (optional)

        Returns:
            List of dictionaries with original and processed images
        """
        # Check if folder exists
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")

        # Get all image files in the folder
        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        image_files.sort()

        if not image_files:
            print(f"No image files found in {folder_path}")
            return []

        results = []

        for filename in image_files:
            try:
                image_path = os.path.join(folder_path, filename)
                result = self.process_image(image_path=image_path)

                # Skip if processing failed
                if 'error' in result:
                    print(f"Skipping {filename}: {result['error']}")
                    continue

                results.append(result)

                # Save processed image if output folder is specified
                if output_folder and result.get('lane_image'):
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    # Save lane image
                    output_path = os.path.join(output_folder, f"lane_{filename}")
                    result['lane_image'].save(output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return results

    def process_numpy_batch(self, batch_images):
        """
        Process a batch of images as numpy arrays.

        Args:
            batch_images: Numpy array of shape (batch_size, height, width, channels)

        Returns:
            Numpy array of lane predictions
        """
        try:
            # Check if input is a numpy array
            if not isinstance(batch_images, np.ndarray):
                raise TypeError("batch_images must be a numpy array")

            # Ensure the input has 4 dimensions (batch, height, width, channels)
            if len(batch_images.shape) != 4:
                raise ValueError("batch_images must have shape (batch_size, height, width, channels)")

            # Resize images if needed
            resized_batch = []
            for img in batch_images:
                pil_img = Image.fromarray(img)
                resized = pil_img.resize((400, 400))
                resized_batch.append(np.array(resized))

            resized_batch = np.array(resized_batch)

            # Get lane predictions
            lane_predictions = self.model(resized_batch)

            # Convert predictions to lane maps
            lanes = np.argmax(lane_predictions, axis=-1) * 255

            return lanes
        except Exception as e:
            print(f"Error processing batch: {e}")
            return None

    def __call__(self, image_or_path):
        """
        Makes the class callable to easily process a single image.

        Args:
            image_or_path: Either a file path or a numpy array

        Returns:
            Lane detection result
        """
        if isinstance(image_or_path, str):
            return self.process_image(image_path=image_or_path)
        elif isinstance(image_or_path, np.ndarray):
            return self.process_image(image_array=image_or_path)
        else:
            raise TypeError("Input must be either a file path (str) or an image array (numpy.ndarray)")


class YOLODetector:
    def __init__(self, model_path):
        """初始化YOLO模型"""
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.last_save_time = 0
        self.save_interval = 2.0  # 保存图片的最小间隔(秒)
        self.detection_threshold = 0.5  # 置信度阈值

        # 创建保存检测结果的目录
        self.save_dir = os.path.join("detections", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.save_dir, exist_ok=True)

        log.debug(f"YOLO模型已加载: {model_path}")
        log.debug(f"可用类别: {self.class_names}")

    def detect(self, frame):
        """对帧进行目标检测"""
        if frame is None:
            return None

        return self.model(frame, conf=self.detection_threshold)[0]


    def process_detections(self, frame, results):
        """处理检测结果并在需要时保存图像"""
        if results is None or frame is None:
            return [], []

        # 提取检测信息
        boxes = results.boxes.cpu().numpy()
        detected_objects = []
        detected_classes = []

        # 处理每个检测结果并绘制标签
        for box in boxes:
            # 提取边界框和类别信息
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]

            # 收集检测结果
            detected_objects.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name
            })
            detected_classes.append(class_name)

            # 绘制边界框和标签
            self._draw_box(frame, x1, y1, x2, y2, class_name, confidence)

        # # 如有检测结果且间隔足够，保存标注后的帧
        # if detected_objects and (time.time() - self.last_save_time) > self.save_interval:
        #     self.save_detection(frame, detected_classes)
        #     self.last_save_time = time.time()

        return detected_objects, detected_classes

    def _draw_box(self, frame, x1, y1, x2, y2, class_name, confidence):
        """在图像上绘制边界框和标签"""
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def save_detection(self, frame, detected_classes):
        """保存检测结果和掩码图像（如果提供）"""
        try:
            # 创建带有时间戳和检测类别的基本文件名
            timestamp = datetime.now().strftime("%H-%M-%S")
            classes_str = "_".join(list(set(detected_classes))) if detected_classes else "no_detection"
            base_filename = f"{timestamp}_{classes_str}"

            # 保存带有检测框的原始图像
            original_filepath = os.path.join(self.save_dir, f"{base_filename}.jpg")
            cv2.imwrite(original_filepath, frame)
            log.debug(f"原始检测结果已保存: {original_filepath}")

        except Exception as e:
            log.error(f"保存检测结果时出错: {str(e)}")
