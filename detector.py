from window import BaseWindow
import cv2
import numpy as np
import time

from ultralytics import YOLO
import os
from datetime import datetime

class RoadDetector(BaseWindow):
    def __init__(self, sx, sy, ex, ey, log=None):
        super().__init__(sx, sy, ex, ey)
        self.lane_center = 0.0  # Position of lane center relative to screen center
        self.road_visible = False
        self.obstacle_detected = False
        self.distance_to_obstacle = 100.0  # meters
        self.log = log

    def process_data(self):
        """Detect lane position and obstacles"""
        if self.color is None:
            return

        try:
            # Convert to more usable formats
            gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(self.color, cv2.COLOR_BGR2HSV)

            # Sample the bottom half of the image for lane detection
            height, width = gray.shape
            bottom_half = gray[height // 2:, :]

            # Simple edge detection
            edges = cv2.Canny(bottom_half, 100, 200)

            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                    minLineLength=50, maxLineGap=10)

            # Lane visibility check
            self.road_visible = lines is not None and len(lines) > 0

            if self.road_visible:
                # Calculate lane center (simplified)
                center_sum = 0
                count = 0

                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    center = (x1 + x2) / 2
                    center_sum += center
                    count += 1

                if count > 0:
                    avg_center = center_sum / count
                    self.lane_center = (avg_center - (width / 2)) / (width / 2)  # -1 to 1 range

            # Simple obstacle detection (looking for large objects in the middle distance)
            # This is highly simplified and would need to be adapted for the actual game
            middle_region = hsv[height // 3:2 * height // 3, width // 3:2 * width // 3]
            # Look for dark objects as potential obstacles
            dark_mask = cv2.inRange(middle_region, (0, 0, 0), (180, 255, 80))

            obstacle_pixels = cv2.countNonZero(dark_mask)
            obstacle_ratio = obstacle_pixels / (middle_region.shape[0] * middle_region.shape[1])

            self.obstacle_detected = obstacle_ratio > 0.2  # Arbitrary threshold

            if self.obstacle_detected:
                # Crude distance calculation
                self.distance_to_obstacle = 100 * (1.0 - obstacle_ratio)
            else:
                self.distance_to_obstacle = 100.0

            self.log.debug(f"Lane center: {self.lane_center:.2f}, Road visible: {self.road_visible}, "
                      f"Obstacle: {self.obstacle_detected}, Distance: {self.distance_to_obstacle:.1f}m")

        except Exception as e:
            self.log.error(f"Error processing road data: {e}")


class YOLODetector:
    def __init__(self, model_path, log=None):
        """Initialize YOLO model for object detection"""
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.last_save_time = 0
        self.save_interval = 2.0  # Minimum seconds between saved images
        self.detection_threshold = 0.5  # Confidence threshold

        # Create directory for saved detections if it doesn't exist
        self.save_dir = os.path.join("detections", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.save_dir, exist_ok=True)
        self.log = log

        self.log.debug(f"YOLO model loaded from {model_path}")
        self.log.debug(f"Available classes: {self.class_names}")

    def detect(self, frame):
        """Run detection on frame and return results"""
        if frame is None:
            return None

        # Run YOLOv8 inference on the frame
        results = self.model(frame, conf=self.detection_threshold)
        return results[0]  # Return first result

    def process_detections(self, frame, results):
        """Process detection results and save image if needed"""
        if results is None or frame is None:
            return [], []

        # Extract detection information
        boxes = results.boxes.cpu().numpy()
        detected_objects = []
        detected_classes = []

        # Process each detection
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

            # Draw bounding box on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the annotated frame if detections present and enough time has passed
        if detected_objects and (time.time() - self.last_save_time) > self.save_interval:
            self.save_detection(frame, detected_classes)
            self.last_save_time = time.time()

        return detected_objects, detected_classes

    def save_detection(self, frame, detected_classes):
        """Save the current frame with detection info"""
        # Create filename with timestamp and detected classes
        timestamp = datetime.now().strftime("%H-%M-%S")
        classes_str = "_".join(list(set(detected_classes)))
        filename = f"{timestamp}_{classes_str}.jpg"
        filepath = os.path.join(self.save_dir, filename)

        # Save the image
        cv2.imwrite(filepath, frame)
        self.log.debug(f"Detection saved: {filepath}")