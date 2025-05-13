# autodrive.py
import multiprocessing as mp
import os

import keyboard
import time
import sys
import signal
import cv2
import numpy as np
from grabscreen import grab_screen, init_camera
from utils.change_window import check_window_resolution_same, correction_window
from window import BaseWindow, set_windows_offset, game_width, game_height
import logging
from pynput.keyboard import Key, Controller

from detector import RoadDetector, YOLODetector, visualize_detections_and_lane

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='autodrive.log',
    filemode='w'
)

running_event = mp.Event()
log = logging.getLogger("AutoDrive")

# Virtual keyboard controller
kb_controller = Controller()

# Control flags
running = True
autodrive_active = False
last_frame_time = 0
target_fps = 30

# Truck control variables
truck_speed = 1.0  # Adjust based on game speed
truck_steering = 0.0  # -1.0 to 1.0 (left to right)
last_key_press_time = 0
key_press_interval = 0.1  # seconds


# Screen regions for detection
# 需要修改检测数据状况，右下角
class TruckDashboard(BaseWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        self.speed = 0.0
        self.rpm = 0.0
        self.gear = 0
        self.fuel = 100.0

    def process_data(self):
        """Extract dashboard information from color data"""
        if self.color is None:
            return

        # Convert to grayscale for processing
        gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

        # Simplified example: detect speed from region
        # In a real implementation, more sophisticated OCR might be needed
        try:
            # Extract speed area
            speed_region = gray[20:60, 30:100]  # Example coordinates

            # Basic thresholding to isolate digits
            _, thresh = cv2.threshold(speed_region, 170, 255, cv2.THRESH_BINARY)

            # For demo purposes: calculate average brightness as a simple "speed" indicator
            brightness = np.mean(thresh)
            self.speed = brightness / 255.0 * 90.0  # Map to 0-90 km/h range

            log.debug(f"Detected speed: {self.speed:.1f} km/h")
        except Exception as e:
            log.error(f"Error processing dashboard: {e}")


# Initialize screen regions
# These coordinates would need to be adjusted for EuroTruck Simulator 2
dashboard = TruckDashboard(100, 500, 400, 600)  # Example coordinates
road_view = RoadDetector(0, 0, game_width, game_height)


def signal_handler(sig, frame):
    global running
    log.info("Exiting...")
    running = False
    sys.exit(0)


def wait_for_game_window(running_event):
    frame = grab_screen()
    if frame is not None and set_windows_offset(frame):
        log.debug("Game window detected and offsets set!")
        return True
    time.sleep(1)
    return False


def press_key(key, duration=0.1):
    """Press and hold a key for a duration"""
    kb_controller.press(key)
    time.sleep(duration)
    kb_controller.release(key)


def control_truck(detected_objects=None):
    """Apply control inputs based on detected objects and lane detection"""
    if not autodrive_active:
        return

    try:
        log.debug("Control truck function called")

        # Default control values
        steer_left = False
        steer_right = False
        need_brake = False
        speed_factor = 0.7  # Default speed

        # Get driving recommendations from lane detection if available
        lane_detected = False
        steering_angle = 0.0
        if hasattr(road_view, 'is_lane_detected') and hasattr(road_view, 'get_drivable_direction'):
            try:
                lane_detected = road_view.is_lane_detected()
                steering_angle = road_view.get_drivable_direction()
                log.debug(f"Lane detected: {lane_detected}, Steering angle: {steering_angle}")

                if hasattr(road_view, 'get_recommended_speed'):
                    speed_factor = road_view.get_recommended_speed()
                    log.debug(f"Recommended speed: {speed_factor}")
            except Exception as e:
                log.error(f"Error getting lane guidance: {e}")

        # Process detected objects for obstacle avoidance
        if detected_objects:
            log.debug(f"Processing {len(detected_objects)} detected objects")

            # Sort objects by proximity (based on bounding box height)
            sorted_objects = sorted(detected_objects,
                                    key=lambda obj: obj['bbox'][3] - obj['bbox'][1],
                                    reverse=True)

            # Group objects by position (left, center, right)
            left_objects = []
            center_objects = []
            right_objects = []

            for obj in sorted_objects:
                if obj['class_name'] in ['car', 'truck', 'bus', 'person', 'bicycle', 'motorcycle']:
                    x1, y1, x2, y2 = obj['bbox']
                    obj_width = x2 - x1
                    obj_height = y2 - y1

                    # Calculate position relative to screen center
                    center_x = (x1 + x2) / 2
                    screen_center_x = game_width / 2
                    relative_pos = (center_x - screen_center_x) / screen_center_x  # -1 to 1

                    # Determine object zone
                    if relative_pos < -0.3:
                        left_objects.append((obj, obj_height, relative_pos))
                    elif relative_pos > 0.3:
                        right_objects.append((obj, obj_height, relative_pos))
                    else:
                        center_objects.append((obj, obj_height, relative_pos))

            log.debug(
                f"Objects by position: left={len(left_objects)}, center={len(center_objects)}, right={len(right_objects)}")

            # based on object positions
            emergency_brake = False

            # Check for immediate collision threats in center
            for obj, height, pos in center_objects:
                # Object directly ahead and close
                if height > game_height / 3.5:
                    emergency_brake = True
                    log.debug(f"Emergency brake - {obj['class_name']} too close ahead")
                    break
                # Object directly ahead but not too close
                elif height > game_height / 5:
                    need_brake = True
                    speed_factor *= 0.5  # Reduce speed by half
                    log.debug(f"Slowing down - {obj['class_name']} ahead")

                    # Decide evasive direction based on position and lane
                    if pos < 0 and not left_objects:
                        steer_left = True
                        log.debug("Evading right to left")
                    elif pos > 0 and not right_objects:
                        steer_right = True
                        log.debug("Evading left to right")

            # If no emergency in center, check left and right
            if not emergency_brake:
                # Check left side objects
                for obj, height, pos in left_objects:
                    if height > game_height / 4:
                        steer_right = True
                        steer_left = False  # Override left steering
                        log.debug(f"Steering right to avoid {obj['class_name']} on left")
                        break

                # Check right side objects
                for obj, height, pos in right_objects:
                    if height > game_height / 4:
                        steer_left = True
                        steer_right = False  # Override right steering
                        log.debug(f"Steering left to avoid {obj['class_name']} on right")
                        break

            # Apply emergency brake if needed
            if emergency_brake:
                log.debug("Applying emergency brake")
                press_key('s', 0.5)
                return  # Exit early to avoid any other controls

        # Apply controls
        log.debug(
            f"Control decision: steer_left={steer_left}, steer_right={steer_right}, lane_steering={steering_angle}, need_brake={need_brake}, speed={speed_factor}")

        # Apply steering
        if steer_left:
            # Steering left from obstacle avoidance
            duration = 0.2
            log.debug(f"Steering left for {duration}s (obstacle avoidance)")
            press_key('a', duration)
        elif steer_right:
            # Steering right from obstacle avoidance
            duration = 0.2
            log.debug(f"Steering right for {duration}s (obstacle avoidance)")
            press_key('d', duration)
        elif lane_detected:
            # Apply lane-based steering
            if steering_angle < -10:
                # Need to steer left
                duration = min(0.1 + abs(steering_angle) / 45.0 * 0.3, 0.4)
                log.debug(f"Lane-based left steering: {steering_angle:.1f} degrees for {duration}s")
                press_key('a', duration)
            elif steering_angle > 10:
                # Need to steer right
                duration = min(0.1 + abs(steering_angle) / 45.0 * 0.3, 0.4)
                log.debug(f"Lane-based right steering: {steering_angle:.1f} degrees for {duration}s")
                press_key('d', duration)

        # Apply speed control
        if need_brake:
            # Apply brakes if needed
            log.debug("Applying brakes")
            press_key('s', 0.2)
        else:
            # Otherwise, accelerate based on recommended speed
            if speed_factor > 0.8:
                # High speed
                duration = 0.3
                log.debug(f"High speed driving for {duration}s")
                press_key('w', duration)
            elif speed_factor > 0.5:
                # Medium speed
                duration = 0.2
                log.debug(f"Medium speed driving for {duration}s")
                press_key('w', duration)
            else:
                # Low speed
                duration = 0.1
                log.debug(f"Low speed driving for {duration}s")
                press_key('w', duration)

    except Exception as e:
        log.error(f"Error in truck control: {e}", exc_info=True)


def update_screen_info():
    """Update all screen information"""
    global last_frame_time

    # Limit update rate
    current_time = time.time()
    if current_time - last_frame_time < 1.0 / target_fps:
        return

    last_frame_time = current_time

    # Grab screen and update window data
    frame = grab_screen()
    if frame is None:
        return

    BaseWindow.set_frame(frame)
    BaseWindow.update_all()

    # Process data from screen regions
    dashboard.process_data()
    road_view.process_data()


def check_keyboard_commands():
    """Check for keyboard commands from user"""
    global autodrive_active

    # Toggle autodrive with F8
    if keyboard.is_pressed('z'):
        autodrive_active = not autodrive_active
        log.debug(f"AutoDrive {'activated' if autodrive_active else 'deactivated'}")
        time.sleep(0.5)  # Debounce

    # Emergency stop with Escape
    if keyboard.is_pressed('x'):
        if autodrive_active:
            autodrive_active = False
            log.debug("Emergency stop activated")
            # Apply brakes
            kb_controller.press('s')
            time.sleep(1.0)
            kb_controller.release('s')


def wait_for_game_window(running_event):
    while running_event.is_set():
        frame = grab_screen()
        if frame is not None and set_windows_offset(frame):
            log.debug("Game window detected and offsets set!")
            return True
        time.sleep(1)
    return False


def main():
    global running, log

    log.info("Starting EuroTruck Simulator 2 AutoDrive")

    # Register signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)

    # Ensure the log directory exists
    os.makedirs("logs", exist_ok=True)

    # Add more detailed logging
    file_handler = logging.FileHandler('logs/autodrive_detailed.log')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    log.debug("Initializing screen capture")
    # Initialize screen capture
    try:
        init_camera(target_fps)
        log.debug("Screen capture initialized successfully")
    except Exception as e:
        log.error(f"Failed to initialize screen capture: {e}")
        return

    try:
        correction_window()
        if not check_window_resolution_same(game_width, game_height):
            log.debug(
                f"Game resolution doesn't match configuration: game_width({game_width}), game_height({game_height})"
            )
            # Continue anyway - we'll work with what we have
    except Exception as e:
        log.debug(f"Warning: Could not set window resolution: {e}")

    running_event.set()
    log.debug(f"Running event set: {running_event.is_set()}")

    # Initialize road detector with U-Net model
    log.debug("Initializing RoadDetector with U-Net model")
    global road_view
    try:
        road_view = RoadDetector(0, 0, game_width, game_height,
                                 model_path="models/lane_detection_model.pth",
                                 log=log)
        log.debug("RoadDetector initialized successfully")
    except Exception as e:
        log.error(f"Failed to initialize RoadDetector: {e}")
        road_view = RoadDetector(0, 0, game_width, game_height, log=log)
        log.debug("Initialized RoadDetector without model")

    # Initialize YOLO detector
    log.debug("Initializing YOLO detector")
    try:
        yolo_detector = YOLODetector("runs/detect/kitti_yolo11/weights/best.pt", log)
        log.debug("YOLO detector initialized successfully")
    except Exception as e:
        log.error(f"Failed to initialize YOLO detector: {e}")
        yolo_detector = None

    # Wait for game window with a timeout
    game_window_found = False
    timeout = 10  # seconds
    start_time = time.time()
    log.debug(f"Waiting for game window (timeout: {timeout}s)")

    if not game_window_found:
        log.debug("Could not detect game window, but continuing with screen capture.")

    log.debug("Press Z to toggle AutoDrive, X for emergency stop")
    print("set", running_event.is_set())

    # Create a window for visualization
    try:
        cv2.namedWindow("AutoDrive Visualization", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AutoDrive Visualization", 800, 600)
        log.debug("Created visualization window")
    except Exception as e:
        log.error(f"Failed to create visualization window: {e}")

    # Add a frame counter and FPS calculation
    frame_count = 0
    start_time = time.time()
    last_fps_print = start_time

    try:
        log.debug("Entering main loop")
        while running_event.is_set():
            # Update FPS counter
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_print >= 5.0:  # Print FPS every 5 seconds
                fps = frame_count / (current_time - start_time)
                log.debug(f"FPS: {fps:.2f}")
                last_fps_print = current_time

            # Check for keyboard commands
            check_keyboard_commands()

            # Get current frame
            frame = grab_screen()
            if wait_for_game_window(running_event) is False:
                log.debug("Failed to capture screen. Retrying...")
                time.sleep(0.5)
                continue

            log.debug(f"Frame captured: {frame.shape}")

            # Process road and lane detection
            if hasattr(road_view, 'process_data'):
                try:
                    road_view.process_data()
                    lane_mask = getattr(road_view, 'lane_mask', None)
                    if lane_mask is not None:
                        log.debug(f"Lane mask: {lane_mask.shape}, lane detected: {road_view._is_lane_detected}")
                    else:
                        log.debug("No lane mask available")
                except Exception as e:
                    log.error(f"Error processing road data: {e}")
                    lane_mask = None
            else:
                log.debug("RoadDetector doesn't have process_data method")
                lane_mask = None

            # Run YOLO detection on frame
            detected_objects = []
            detected_classes = []
            if yolo_detector is not None:
                try:
                    results = yolo_detector.detect(frame)
                    detected_objects, detected_classes = yolo_detector.process_detections(frame, results)
                    log.debug(f"Detected {len(detected_objects)} objects: {detected_classes}")
                except Exception as e:
                    log.error(f"Error in YOLO detection: {e}")

            # Visualize detections and lane mask
            try:
                visualize_detections_and_lane(log, road_view, frame, detected_objects, lane_mask)
                log.debug("Visualization updated")
            except Exception as e:
                log.error(f"Error in visualization: {e}")

            # Display status information
            try:
                status_frame = np.zeros((200, 400, 3), dtype=np.uint8)
                cv2.putText(status_frame, f"AutoDrive: {'ON' if autodrive_active else 'OFF'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if autodrive_active else (0, 0, 255),
                            2)

                if hasattr(road_view, '_is_lane_detected') and road_view._is_lane_detected:
                    cv2.putText(status_frame, f"Lane Detected: YES",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(status_frame, f"Lane Offset: {road_view._lane_center_offset:.2f}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(status_frame, f"Curvature: {road_view._lane_curvature:.2f}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if hasattr(road_view, 'get_drivable_direction'):
                        cv2.putText(status_frame, f"Steering: {road_view.get_drivable_direction():.1f} deg",
                                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if hasattr(road_view, 'get_recommended_speed'):
                        cv2.putText(status_frame, f"Speed: {road_view.get_recommended_speed():.2f}",
                                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(status_frame, f"Lane Detected: NO",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("AutoDrive Status", status_frame)
                log.debug("Status display updated")
            except Exception as e:
                log.error(f"Error updating status display: {e}")

            # Control truck based on detections and lane information
            if autodrive_active:
                try:
                    control_truck(detected_objects)
                    log.debug("Truck control applied")
                except Exception as e:
                    log.error(f"Error in truck control: {e}")

            # Check for exit command
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                log.debug("ESC key pressed, exiting main loop")
                running_event.clear()
                break

            # Short sleep to reduce CPU usage
            time.sleep(0.01)  # Use a shorter sleep time to make it more responsive

    except KeyboardInterrupt:
        log.info("Main process: Exiting due to keyboard interrupt...")
    except Exception as e:
        log.error(f"An error occurred in main: {e}", exc_info=True)
    finally:
        # Release all held keys
        for key in ['w', 'a', 's', 'd']:
            kb_controller.release(key)
        print("Released all keys")
        log.info("Released all keys")
        cv2.destroyAllWindows()
        running_event.clear()
        log.info("Main loop exited, running_event cleared")


if __name__ == '__main__':
    main()