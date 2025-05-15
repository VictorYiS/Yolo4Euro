import os
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from log import log
from road_lane_segmentation import UNet


class RoadDetector():
    def __init__(self, model_path="models/lane_detection_model.pth"):
        self.lane_center = 0.0  # Position of lane center relative to screen center
        self.road_visible = False
        self.obstacle_detected = False
        self.distance_to_obstacle = 100.0  # meters
        self.lane_mask = None
        self._lane_center_offset = 0.0
        self._lane_curvature = 0.0
        self._is_lane_detected = False

        # Load the U-Net model
        self.load_model(model_path)

        # Preprocessing parameters for the model
        self.input_size = (512, 256)  # Width, Height - match the model's expected input
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        log.debug("RoadDetector initialized with U-Net model")

    def load_model(self, model_path):
        """Load the pre-trained U-Net model"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log.debug(f"Using device: {self.device}")

            # Check if the model file exists
            model_exists = os.path.exists(model_path)
            log.debug(f"Model path: {os.path.abspath(model_path)}, exists: {model_exists}")

            # Initialize model
            log.debug("Initializing U-Net model...")
            try:
                self.model = UNet(in_channels=3, out_channels=1)
                self.model.float()  # Explicitly set model to float32
                log.debug("U-Net model initialized")
            except Exception as e:
                log.error(f"Failed to initialize U-Net model: {e}")
                self.model = None
                return

            # Load model weights if file exists
            if model_exists:
                try:
                    log.debug(f"Loading model weights from {model_path}")
                    # Try loading with map_location first
                    checkpoint = torch.load(model_path, map_location=self.device)
                    log.debug(
                        f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'not a dict'}")

                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        log.debug("Loaded model weights from 'model_state_dict'")
                    elif isinstance(checkpoint, dict):
                        # Maybe it's just the state dict directly
                        self.model.load_state_dict(checkpoint)
                        log.debug("Loaded model weights directly from checkpoint")
                    else:
                        log.error("Checkpoint is not a valid format")
                        # Continue with random weights

                    self.model = self.model.to(self.device)
                    self.model.eval()  # Set model to evaluation mode
                    log.debug(f"Successfully loaded model from {model_path}")
                except Exception as e:
                    log.error(f"Error loading model state dict: {e}", exc_info=True)
                    log.debug("Using model with random weights")
                    # Continue with random weights
                    self.model = self.model.to(self.device)
                    self.model.eval()
            else:
                log.warning(f"Model file not found at {model_path}")
                log.debug("Using model with random weights")
                # Continue with random weights
                self.model = self.model.to(self.device)
                self.model.eval()

            # Test the model with a dummy input to see if it works
            try:
                log.debug("Testing model with dummy input...")
                dummy_input = torch.randn(1, 3, 256, 512).to(self.device)
                with torch.no_grad():
                    dummy_output = self.model(dummy_input)
                log.debug(f"Dummy output shape: {dummy_output.shape}")
                log.debug("Model test successful")
            except Exception as e:
                log.error(f"Model test failed: {e}", exc_info=True)
                self.model = None

        except Exception as e:
            log.error(f"Error in load_model: {e}", exc_info=True)
            self.model = None

    def preprocess_image(self, image):
        """Preprocess image for U-Net model input"""
        try:
            # Resize to model input dimensions
            img_resized = cv2.resize(image, self.input_size)

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # Normalize
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_normalized = (img_normalized - self.mean) / self.std

            # Convert HWC to CHW (channels first for PyTorch)
            img_chw = img_normalized.transpose(2, 0, 1)

            # Create tensor and add batch dimension
            input_tensor = torch.from_numpy(img_chw).float().unsqueeze(0)

            return input_tensor
        except Exception as e:
            log.error(f"Error preprocessing image: {e}")
            return None

    def _use_color_based_detection(self, frame):
        """Fallback method using color thresholding for lane detection"""
        try:
            # Convert to your preferred color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # In EuroTruck, road lanes can be white, yellow, or even gray
            # Values for white lines (more permissive)
            lower_white = np.array([0, 0, 150])
            upper_white = np.array([180, 60, 255])

            # Values for yellow lines (more permissive)
            lower_yellow = np.array([15, 80, 80])
            upper_yellow = np.array([35, 255, 255])

            # Values for gray lines (can be common in games)
            lower_gray = np.array([0, 0, 100])
            upper_gray = np.array([180, 30, 180])

            # Create masks
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)

            # Combine masks
            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
            combined_mask = cv2.bitwise_or(combined_mask, gray_mask)

            # Focus on the bottom part of the image (where lanes are more likely)
            h, w = combined_mask.shape
            roi_mask = np.zeros_like(combined_mask)
            roi_mask[h // 2:, :] = 255  # Only keep bottom half
            combined_mask = cv2.bitwise_and(combined_mask, roi_mask)

            # Apply some morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            # Try to detect lane-like shapes using Hough lines
            # This helps remove random noise
            refined_mask = np.zeros_like(combined_mask)
            try:
                lines = cv2.HoughLinesP(combined_mask, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=100)
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(refined_mask, (x1, y1), (x2, y2), 255, 10)

                    # Apply dilation to connect nearby lines
                    refined_mask = cv2.dilate(refined_mask, kernel, iterations=2)
                    self.lane_mask = refined_mask
                    log.debug(f"Lane detection with Hough transform: {len(lines)} lines detected")
                else:
                    # Fallback to the combined mask if no lines detected
                    self.lane_mask = combined_mask
                    log.debug("No lines detected with Hough transform, using combined mask")
            except Exception as e:
                log.error(f"Error in Hough transform: {e}")
                # Fallback to the combined mask
                self.lane_mask = combined_mask

            # Count lane pixels for debugging
            lane_pixels = np.sum(self.lane_mask > 0)
            total_pixels = self.lane_mask.size
            log.debug(
                f"Color-based lane detection: {lane_pixels}/{total_pixels} pixels ({lane_pixels / total_pixels * 100:.2f}%)")
        except Exception as e:
            log.error(f"Error in color-based lane detection: {e}")
            self.lane_mask = None

    def process_data(self, frame):
        """Process road view to detect lanes"""
        if frame is None:
            log.debug("No color data available")
            return

        try:
            # For debugging, log the color data dimensions
            log.debug(f"Processing color data with shape: {frame.shape}")

            # Try U-Net model first
            if self.model is not None:
                try:
                    log.debug("Attempting to use U-Net model")
                    # Create a preprocessed input for the model
                    input_tensor = self.preprocess_image(frame)

                    if input_tensor is not None:
                        # Run inference
                        input_tensor = input_tensor.to(self.device)
                        with torch.no_grad():
                            prediction = self.model(input_tensor)
                            prediction = prediction.squeeze().cpu().numpy()

                        # Convert prediction to binary mask
                        self.lane_mask = (prediction > 0.5).astype(np.uint8) * 255

                        # Resize to original frame size
                        self.lane_mask = cv2.resize(self.lane_mask, (frame.shape[1], frame.shape[0]))
                        log.debug(f"U-Net lane detection completed, mask shape: {self.lane_mask.shape}")
                    else:
                        log.warning("Failed to preprocess image for U-Net")
                        self._use_color_based_detection(frame)
                except Exception as e:
                    log.error(f"Error using U-Net model: {e}", exc_info=True)
                    self._use_color_based_detection(frame)
            else:
                log.debug("No U-Net model available, using color-based detection")
                self._use_color_based_detection(frame)

            # For debugging, visualize the lane mask
            if self.lane_mask is not None:
                # Calculate pixel statistics for debugging
                white_pixels = np.sum(self.lane_mask > 0)
                total_pixels = self.lane_mask.size
                percentage = (white_pixels / total_pixels) * 100
                log.debug(f"Lane mask statistics: {white_pixels}/{total_pixels} white pixels ({percentage:.2f}%)")

                # Save lane mask for debugging
                try:
                    debug_dir = "debug_images"
                    os.makedirs(debug_dir, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite(f"{debug_dir}/lane_mask_{timestamp}.jpg", self.lane_mask)
                    log.debug(f"Saved lane mask to {debug_dir}/lane_mask_{timestamp}.jpg")
                except Exception as e:
                    log.error(f"Failed to save debug lane mask: {e}")
            else:
                log.warning("No lane mask was generated")

            # Calculate lane metrics
            self._lane_center_offset = self.get_lane_center_offset()
            self._lane_curvature = self.get_lane_curvature()
            self._is_lane_detected = self.is_lane_detected()

            log.debug(f"Lane detection results: detected={self._is_lane_detected}, " +
                           f"offset={self._lane_center_offset:.2f}, curvature={self._lane_curvature:.2f}")

        except Exception as e:
            log.error(f"Error in process_data: {e}", exc_info=True)
            self.lane_mask = None
            self._lane_center_offset = 0.0
            self._lane_curvature = 0.0
            self._is_lane_detected = False

    def _use_color_based_detection(self, frame):
        """Fallback method using color thresholding for lane detection"""
        try:
            # Convert to your preferred color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Adjust these values based on the lane colors in the game
            # Values for white lines
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])

            # Values for yellow lines
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])

            # Create masks for white and yellow lines
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # Combine masks
            self.lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

            # Apply some morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            self.lane_mask = cv2.morphologyEx(self.lane_mask, cv2.MORPH_CLOSE, kernel)
            self.lane_mask = cv2.morphologyEx(self.lane_mask, cv2.MORPH_OPEN, kernel)

            log.debug("Color-based lane detection applied")
        except Exception as e:
            log.error(f"Error in color-based lane detection: {e}")
            self.lane_mask = None

    def create_visualization(self, frame):
        """Create a visualization of the lane detection for debugging"""
        if frame is None or self.lane_mask is None:
            return None

        try:
            # Create a colored overlay for visualization
            vis_frame = frame.copy()
            lane_overlay = np.zeros_like(vis_frame)
            lane_overlay[:, :, 1] = self.lane_mask  # Green channel

            # Blend the overlay with the original image
            alpha = 0.4  # Transparency factor
            vis_frame = cv2.addWeighted(vis_frame, 1, lane_overlay, alpha, 0)

            # Add text with lane metrics
            cv2.putText(vis_frame, f"Lane Offset: {self._lane_center_offset:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Curvature: {self._lane_curvature:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Lane Detected: {self._is_lane_detected}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            return vis_frame
        except Exception as e:
            log.error(f"Error creating visualization: {e}")
            return None

    def get_lane_center_offset(self):
        """
        Returns the offset from the center of the lane
        Negative values mean the vehicle is left of center
        Positive values mean the vehicle is right of center
        Range: -1.0 to 1.0
        """
        try:
            if self.lane_mask is not None:
                height, width = self.lane_mask.shape[:2]

                # Focus on the bottom third of the image (closer to the vehicle)
                bottom_section = self.lane_mask[int(height * 2 / 3):, :]

                # Find all lane pixels
                lane_pixels = np.where(bottom_section > 0)

                if len(lane_pixels[1]) > 0:
                    # Calculate average x position of lane pixels
                    lane_center_x = np.mean(lane_pixels[1])

                    # Calculate offset from center (-1 to 1)
                    screen_center_x = width / 2
                    offset = (lane_center_x - screen_center_x) / screen_center_x

                    # Apply some smoothing to avoid jittery steering
                    offset = np.clip(offset, -1.0, 1.0)

                    return offset

            return 0.0  # Default: assume centered
        except Exception as e:
            log.error(f"Error calculating lane offset: {e}")
            return 0.0

    def get_lane_curvature(self):
        """
        Returns a measure of lane curvature
        Higher values indicate sharper curves
        Range: 0.0 (straight) to 1.0 (very curved)
        """
        try:
            if self.lane_mask is not None:
                height, width = self.lane_mask.shape[:2]

                # Divide image into 3 vertical sections from bottom to top
                section_height = height // 3

                sections = [
                    self.lane_mask[height - section_height:height, :],  # Bottom
                    self.lane_mask[height - 2 * section_height:height - section_height, :],  # Middle
                    self.lane_mask[height - 3 * section_height:height - 2 * section_height, :]  # Top
                ]

                # Find center of lane in each section
                centers = []
                for i, section in enumerate(sections):
                    lane_pixels = np.where(section > 0)
                    if len(lane_pixels[1]) > 0:
                        center_x = np.mean(lane_pixels[1])
                        centers.append(center_x)
                    else:
                        # If we don't find lane pixels in a section, use previous section's center
                        # or the screen center if it's the first section
                        centers.append(centers[-1] if centers else (width / 2))

                if len(centers) >= 3:
                    # Calculate how much the center position changes (lane curvature)
                    # Weight the bottom section more (closer to vehicle)
                    delta1 = abs(centers[0] - centers[1]) / width  # Bottom to middle
                    delta2 = abs(centers[1] - centers[2]) / width  # Middle to top

                    # Combine with more weight to the closer section
                    curvature = (delta1 * 0.7 + delta2 * 0.3)

                    # Normalize to 0-1 range with scaling factor
                    return min(curvature * 5.0, 1.0)  # Scale factor can be adjusted

            return 0.0  # Default: assume straight road
        except Exception as e:
            log.error(f"Error calculating curvature: {e}")
            return 0.0

    def is_lane_detected(self):
        """
        Returns whether lane markings were successfully detected
        """
        try:
            if self.lane_mask is not None:
                # We need to have enough lane pixels to consider the lane detected
                # Reduce the threshold to make it more sensitive
                min_pixels = self.lane_mask.shape[0] * self.lane_mask.shape[1] * 0.01  # Only 1% of pixels
                detected_pixels = np.sum(self.lane_mask > 0)

                # Log actual values for debugging
                log.debug(f"Lane pixels: {detected_pixels}/{self.lane_mask.size}, min required: {min_pixels}")

                # Consider lane detected if there are enough pixels
                return detected_pixels > min_pixels

            return False
        except Exception as e:
            log.error(f"Error checking lane detection: {e}")
            return False

    def get_drivable_direction(self):
        """
        Returns a recommended driving direction based on lane detection
        Returns: angle in degrees, where 0 is straight ahead,
                negative is left, positive is right
        """
        try:
            if not self.is_lane_detected():
                return 0.0  # If no lane detected, suggest going straight

            # Convert offset to steering angle
            # Scale factor determines how aggressively to steer
            # Higher curvature should result in more aggressive steering
            scale_factor = 1.0 + self.get_lane_curvature() * 2.0  # Range: 1.0 to 3.0

            # Convert offset (-1 to 1) to angle (-30 to 30 degrees)
            # Apply scaling based on curvature
            steering_angle = -self._lane_center_offset * 30.0 * scale_factor

            # Limit maximum steering angle
            steering_angle = np.clip(steering_angle, -45.0, 45.0)

            return steering_angle
        except Exception as e:
            log.error(f"Error calculating drivable direction: {e}")
            return 0.0

    def get_recommended_speed(self):
        """
        Returns a recommended speed factor based on road conditions
        Range: 0.0 (stop) to 1.0 (full speed)
        """
        try:
            if not self.is_lane_detected():
                return 0.5  # Moderate speed if no lane detected

            # Reduce speed for curves
            curvature = self.get_lane_curvature()

            # Sharper curves = slower speed (inverse relationship)
            # speed_factor ranges from 0.4 to 1.0
            speed_factor = 1.0 - (curvature * 0.6)

            return speed_factor
        except Exception as e:
            log.error(f"Error calculating recommended speed: {e}")
            return 0.5  # Default to moderate speed


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


    def process_detections(self, frame, results, roi, lane_mask_detector):
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

        # 如有检测结果且间隔足够，保存标注后的帧
        if detected_objects and (time.time() - self.last_save_time) > self.save_interval:
            self.save_detection(frame, detected_classes, roi, lane_mask_detector)
            self.last_save_time = time.time()

        return detected_objects, detected_classes

    def _draw_box(self, frame, x1, y1, x2, y2, class_name, confidence):
        """在图像上绘制边界框和标签"""
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def save_detection(self, frame, detected_classes, roi, lane_mask_detector):
        """保存检测结果和掩码图像（如果提供）"""
        # 创建带有时间戳和检测类别的基本文件名
        timestamp = datetime.now().strftime("%H-%M-%S")
        classes_str = "_".join(list(set(detected_classes)))
        base_filename = f"{timestamp}_{classes_str}"

        # 保存带有检测框的原始图像
        original_filepath = os.path.join(self.save_dir, f"{base_filename}.jpg")
        cv2.imwrite(original_filepath, frame)
        log.debug(f"原始检测结果已保存: {original_filepath}")

        lane_mask_detector.process_data(roi)
        lane_mask_detector.create_visualization(roi)
