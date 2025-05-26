import os
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from log import log
from train_e_net import ENet  # Import ENet class


class LaneDetector:
    """
    Lane detection class that uses an E-Net model to detect lane lines in images.
    """

    def __init__(self, model_path='models/ENET.pth', use_gpu=True):
        """
        Initialize the lane detector with the E-Net model.

        Args:
            model_path: Path to the E-Net model weights
            use_gpu: Whether to use GPU for inference
        """
        # Configure device
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        # Initialize E-Net model
        self.model = ENet(2, 4)  # Binary and instance segmentation

        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print(f"E-Net model loaded from {model_path} using {self.device}")

    def process_image(self, image_path=None, image_array=None, output_size=(1000, 500)):
        """
        Process an image to detect lane lines with E-Net.

        Args:
            image_path: Path to image file
            image_array: Image as numpy array (RGB)
            output_size: Size to stretch output (width, height)

        Returns:
            Dictionary with original image, processed lane image and data
        """
        try:
            # Load and prepare image
            if image_path is not None:
                image = Image.open(image_path).convert('RGB')
                img_array = np.array(image)
            elif image_array is not None:
                img_array = image_array
                image = Image.fromarray(img_array.astype('uint8'))
            else:
                raise ValueError("Must provide either image_path or image_array")

            # Convert to grayscale and resize to model input size (512x256)
            if len(img_array.shape) == 3:
                gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                # Already grayscale
                gray_img = img_array

            resized_img = cv2.resize(gray_img, (512, 256))
            # Add channel dimension
            model_input = resized_img[..., np.newaxis]

            # Convert to tensor
            input_tensor = torch.from_numpy(model_input).float().permute(2, 0, 1)
            input_tensor = input_tensor.unsqueeze(0).to(self.device)

            # Get model predictions
            with torch.no_grad():
                binary_logits, instance_logits = self.model(input_tensor)

            # Process predictions
            binary_seg = torch.argmax(binary_logits, dim=1).squeeze().cpu().numpy()
            instance_seg = torch.argmax(instance_logits, dim=1).squeeze().cpu().numpy()

            # Ensure arrays are the correct type before resizing
            binary_seg_uint8 = binary_seg.astype(np.uint8) * 255
            instance_seg_uint8 = instance_seg.astype(np.uint8)

            # Resize outputs to desired dimensions
            binary_output = cv2.resize(binary_seg_uint8, output_size, interpolation=cv2.INTER_NEAREST)
            instance_output = cv2.resize(instance_seg_uint8, output_size, interpolation=cv2.INTER_NEAREST)

            # Create lane image from binary segmentation
            lane_image = Image.fromarray(binary_output, 'L')

            return {
                'original_image': image,
                'lane_image': lane_image,
                'lane_data': binary_output,
                'instance_data': instance_output
            }
        except Exception as e:
            print(f"Error processing image: {e}")
            return {'error': str(e)}

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
    def __init__(self, model_path, device='cuda:0'):
        """Initialize the YOLO detector with the specified model path and device."""
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        self.class_names = self.model.names
        self.last_save_time = 0
        self.save_interval = 2.0  # save interval in seconds
        self.detection_threshold = 0.5  # confidence threshold for detections

        log.debug(f"YOLO loaded: {model_path}")
        log.debug(f"classes: {self.class_names}")

    def detect(self, frame):
        """detect objects in the given frame using the YOLO model."""
        if frame is None:
            return None

        return self.model(frame, conf=self.detection_threshold, device=self.device)[0]


    def process_detections(self, frame, results):
        """process the detection results and draw bounding boxes on the frame."""
        if results is None or frame is None:
            return [], []

        # extract bounding boxes and class IDs from results
        boxes = results.boxes.cpu().numpy()
        detected_objects = []
        detected_classes = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]

            detected_objects.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name
            })
            detected_classes.append(class_name)

            self._draw_box(frame, x1, y1, x2, y2, class_name, confidence)

        # if detected_objects and (time.time() - self.last_save_time) > self.save_interval:
        #     self.save_detection(frame, detected_classes)
        #     self.last_save_time = time.time()

        return detected_objects, detected_classes

    def _draw_box(self, frame, x1, y1, x2, y2, class_name, confidence):
        """draw a bounding box and label on the frame."""
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def save_detection(self, frame, detected_classes):
        """save the detection results to a file."""
        try:
            # wait for the save interval to avoid frequent saves
            timestamp = datetime.now().strftime("%H-%M-%S")
            classes_str = "_".join(list(set(detected_classes))) if detected_classes else "no_detection"
            base_filename = f"{timestamp}_{classes_str}"

            # save the frame with detections
            original_filepath = os.path.join(self.save_dir, f"{base_filename}.jpg")
            cv2.imwrite(original_filepath, frame)
            log.debug(f"original file saved: {original_filepath}")

        except Exception as e:
            log.error(f"error in saving detection: {str(e)}")
