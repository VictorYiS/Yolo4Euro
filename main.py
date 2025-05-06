# autodrive.py
import multiprocessing as mp
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

from detector import RoadDetector, YOLODetector

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
    while running_event.is_set():
        frame = grab_screen()
        if frame is not None and set_windows_offset(frame):
            log.debug("Game window detected and offsets set!")
            return True
        time.sleep(1)
    return False


# def process_yolo_detection(detector):
#     """Process YOLO detection on current frame"""
#     try:
#         # Get current frame
#         frame = grab_screen()
#         if frame is None:
#             return
#
#         # Run detection
#         results = detector.detect(frame)
#
#         # Process and visualize detections
#         detected_objects, detected_classes = detector.process_detections(frame, results)
#
#         if detected_objects:
#             log.debug(f"Detected {len(detected_objects)} objects: {detected_classes}")
#
#             # You can add additional logic here based on detections
#             # For example, emergency braking for obstacles
#             for obj in detected_objects:
#                 # Example: If a vehicle is detected too close, brake
#                 if obj['class_name'] in ['car', 'truck', 'bus', 'van'] and obj['confidence'] > 0.7:
#                     # Calculate position and size to determine proximity
#                     x1, y1, x2, y2 = obj['bbox']
#                     box_height = y2 - y1
#
#                     # If object is in bottom half of screen and large, it's probably close
#                     if y2 > frame.shape[0] / 2 and box_height > frame.shape[0] / 4:
#                         log.warning(f"Emergency braking! Large {obj['class_name']} detected")
#                         if autodrive_active:
#                             press_key('s', 0.5)  # Apply brakes
#
#     except Exception as e:
#         log.error(f"Error in YOLO detection: {e}")


def press_key(key, duration=0.1):
    """Press and hold a key for a duration"""
    kb_controller.press(key)
    time.sleep(duration)
    kb_controller.release(key)


def control_truck(detected_objects=None):
    """Apply control inputs based on detected objects"""
    if not autodrive_active:
        return

    try:
        # Default movement - go forward
        if not detected_objects:
            press_key('w', 0.2)
            return

        # Check for objects that need avoidance
        need_braking = False
        need_left_turn = False
        need_right_turn = False

        for obj in detected_objects:
            if obj['class_name'] in ['car', 'truck', 'bus', 'person', 'bicycle', 'motorcycle']:
                x1, y1, x2, y2 = obj['bbox']
                obj_width = x2 - x1
                obj_height = y2 - y1

                # Calculate position relative to screen center
                center_x = (x1 + x2) / 2
                screen_center_x = window.game_width / 2
                relative_pos = (center_x - screen_center_x) / screen_center_x  # -1 to 1

                # Check if object is too close (large height)
                if obj_height > window.game_height / 4:
                    need_braking = True
                # If object is on left side but not too close
                elif relative_pos < -0.2 and obj_height > window.game_height / 6:
                    need_right_turn = True
                # If object is on right side but not too close
                elif relative_pos > 0.2 and obj_height > window.game_height / 6:
                    need_left_turn = True

        # Apply controls based on detection
        if need_braking:
            press_key('s', 0.3)
        elif need_left_turn:
            press_key('a', 0.2)
        elif need_right_turn:
            press_key('d', 0.2)
        else:
            press_key('w', 0.2)

    except Exception as e:
        log.debug(f"Error in truck control: {e}")


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


def main():
    global running

    log.info("Starting EuroTruck Simulator 2 AutoDrive")

    # Register signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize screen capture
    init_camera(target_fps)

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

    # Wait for game window with a timeout
    game_window_found = False
    timeout = 10  # seconds
    start_time = time.time()

    while time.time() - start_time < timeout:
        frame = grab_screen()
        if frame is not None:
            # Just use the frame as-is if window detection fails
            BaseWindow.set_frame(frame)
            BaseWindow.update_all()
            game_window_found = True
            break
        time.sleep(1)

    if not game_window_found:
        log.debug("Could not detect game window, but continuing with screen capture.")

    # Initialize YOLO detector
    yolo_detector = YOLODetector("runs/detect/kitti_yolo11/weights/best.pt", log)

    log.debug("Press Z to toggle AutoDrive, ESC for emergency stop")
    print("set",running_event.is_set())

    try:
        while running_event.is_set():
            # Check for keyboard commands
            check_keyboard_commands()

            # Get current frame
            frame = grab_screen()
            if frame is None:
                log.debug("Failed to capture screen. Retrying...")
                time.sleep(0.5)
                continue

            # Update window with current frame
            BaseWindow.set_frame(frame)
            BaseWindow.update_all()

            # Run YOLO detection on frame
            results = yolo_detector.detect(frame)
            detected_objects, detected_classes = yolo_detector.process_detections(frame, results)

            # # Visualize detections (can be commented out if not needed)
            # visualize_detections(frame, detected_objects)

            # Control truck based on detections
            if autodrive_active:
                control_truck(detected_objects)

            # Short sleep to reduce CPU usage
            time.sleep(0.05)

    except KeyboardInterrupt:
        log.debug("Main process: Exiting due to keyboard interrupt...")
    except Exception as e:
        log.debug(f"An error occurred in main: {e}")
    finally:
        # Release all held keys
        for key in ['w', 'a', 's', 'd']:
            kb_controller.release(key)
        print("Released all keys")
        cv2.destroyAllWindows()
        running_event.clear()


if __name__ == '__main__':
    main()