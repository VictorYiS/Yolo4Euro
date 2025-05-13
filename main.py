# autodrive.py - Refactored Euro Truck Simulator 2 autodrive system
import os
import sys
import time
import signal
import logging
import cv2
import numpy as np
from grabscreen import grab_screen, init_camera
from utils.change_window import check_window_resolution_same, correction_window
from window import BaseWindow, set_windows_offset, game_width, game_height
from pynput import keyboard
from detector import RoadDetector, YOLODetector, visualize_detections_and_lane

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO to reduce log volume
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='autodrive.log',
    filemode='w'
)
log = logging.getLogger("AutoDrive")


# Global state management
class State:
    running = True
    autodrive_active = False
    last_frame_time = 0
    target_fps = 30


# Keyboard controller setup
class KeyController:
    def __init__(self):
        self.controller = keyboard.Controller()
        self.key_press_interval = 0.05  # Reduced from 0.1 for more responsive controls
        self.last_press_time = {}  # Track last press time for each key

        # Set up keyboard listener for toggle commands
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

    def on_key_press(self, key):
        try:
            if key.char == 'z':
                # Toggle autodrive
                State.autodrive_active = not State.autodrive_active
                log.info(f"AutoDrive {'activated' if State.autodrive_active else 'deactivated'}")
            elif key.char == 'x':
                # Emergency stop
                if State.autodrive_active:
                    State.autodrive_active = False
                    log.info("Emergency stop activated")
                    self.press_key('s', 1.0)  # Apply brakes
            elif key == keyboard.Key.esc:
                # Exit program
                State.running = False
                log.info("ESC pressed, exiting program")
                return False  # Stop listener
        except AttributeError:
            # Special key handling
            pass

        return True  # Continue listening

    def press_key(self, key, duration=0.1):
        """Press and hold a key for a specific duration"""
        current_time = time.time()

        # Rate limiting to prevent excessive key presses
        if key in self.last_press_time and current_time - self.last_press_time.get(key, 0) < self.key_press_interval:
            return

        try:
            self.controller.press(key)
            time.sleep(duration)
            self.controller.release(key)
            self.last_press_time[key] = current_time
            log.debug(f"Pressed key {key} for {duration}s")
        except Exception as e:
            log.error(f"Error pressing key {key}: {e}")


# Simplified truck dashboard
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

        # Basic implementation - can be expanded with OCR for actual values
        try:
            gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)
            speed_region = gray[20:60, 30:100]  # Example coordinates
            _, thresh = cv2.threshold(speed_region, 170, 255, cv2.THRESH_BINARY)
            brightness = np.mean(thresh)
            self.speed = brightness / 255.0 * 90.0  # Map to 0-90 km/h range
        except Exception as e:
            log.error(f"Error processing dashboard: {e}")


# Truck controller with simplified logic
class TruckController:
    def __init__(self, keyboard_controller):
        self.kb = keyboard_controller

    def control(self, road_detector, detected_objects=None):
        """Apply control inputs based on detected objects and lane detection"""
        if not State.autodrive_active:
            return

        try:
            # Default control values
            steer_left = False
            steer_right = False
            need_brake = False
            speed_factor = 0.7  # Default speed

            # Get lane information if available
            steering_angle = 0.0
            lane_detected = False

            if hasattr(road_detector, 'is_lane_detected') and road_detector.is_lane_detected():
                lane_detected = True
                steering_angle = road_detector.get_drivable_direction()

                if hasattr(road_detector, 'get_recommended_speed'):
                    speed_factor = road_detector.get_recommended_speed()

            # Process detected objects for obstacle avoidance
            if detected_objects:
                # Check for immediate obstacles in front
                for obj in detected_objects:
                    if obj['class_name'] in ['car', 'truck', 'bus', 'person', 'bicycle', 'motorcycle']:
                        x1, y1, x2, y2 = obj['bbox']
                        obj_height = y2 - y1
                        center_x = (x1 + x2) / 2
                        screen_center_x = game_width / 2
                        relative_pos = (center_x - screen_center_x) / screen_center_x  # -1 to 1

                        # Object directly ahead and close
                        if abs(relative_pos) < 0.3 and obj_height > game_height / 4:
                            need_brake = True

                            # Emergency brake for very close objects
                            if obj_height > game_height / 3:
                                self.kb.press_key('s', 0.5)
                                return  # Exit early

                            # Evasive steering based on position
                            if relative_pos < 0:
                                steer_right = True
                            else:
                                steer_left = True

                        # Objects on sides - avoid if close
                        elif relative_pos < -0.3 and obj_height > game_height / 5:  # Left side
                            steer_right = True
                        elif relative_pos > 0.3 and obj_height > game_height / 5:  # Right side
                            steer_left = True

            # Apply lane-based steering if no obstacles requiring evasion
            if lane_detected and not (steer_left or steer_right):
                if steering_angle < -10:
                    # Need to steer left
                    duration = min(0.1 + abs(steering_angle) / 45.0 * 0.2, 0.3)
                    self.kb.press_key('a', duration)
                elif steering_angle > 10:
                    # Need to steer right
                    duration = min(0.1 + abs(steering_angle) / 45.0 * 0.2, 0.3)
                    self.kb.press_key('d', duration)
            elif steer_left:
                self.kb.press_key('a', 0.2)
            elif steer_right:
                self.kb.press_key('d', 0.2)

            # Apply speed control
            if need_brake:
                self.kb.press_key('s', 0.2)
            else:
                # Accelerate based on recommended speed
                duration = min(0.1 + speed_factor * 0.2, 0.3)
                self.kb.press_key('w', duration)

        except Exception as e:
            log.error(f"Error in truck control: {e}")


# Main application class
class AutoDrive:
    def __init__(self):
        self.init_system()
        self.key_controller = KeyController()
        self.truck_controller = TruckController(self.key_controller)
        self.dashboard = TruckDashboard(100, 500, 400, 600)  # Example coordinates
        self.road_detector = None
        self.yolo_detector = None
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_print = self.start_time

    def init_system(self):
        """Initialize the autodrive system"""
        log.info("Starting EuroTruck Simulator 2 AutoDrive")

        # Register signal handler for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)

        # Ensure log directory exists
        os.makedirs("logs", exist_ok=True)

        # Initialize screen capture
        try:
            init_camera(State.target_fps)
            log.info("Screen capture initialized")
        except Exception as e:
            log.error(f"Failed to initialize screen capture: {e}")
            sys.exit(1)

        # Check window resolution
        try:
            correction_window()
            if not check_window_resolution_same(game_width, game_height):
                log.warning(f"Game resolution doesn't match configuration: {game_width}x{game_height}")
        except Exception as e:
            log.warning(f"Could not set window resolution: {e}")

    def init_detectors(self):
        """Initialize the road and object detectors"""
        # Initialize road detector
        try:
            self.road_detector = RoadDetector(
                0, 0, game_width, game_height,
                model_path="models/lane_detection_model.pth",
                log=log
            )
            log.info("Road detector initialized")
        except Exception as e:
            log.error(f"Failed to initialize road detector: {e}")
            self.road_detector = RoadDetector(0, 0, game_width, game_height, log=log)

        # Initialize YOLO detector
        try:
            self.yolo_detector = YOLODetector("runs/detect/kitti_yolo11/weights/best.pt", log)
            log.info("YOLO detector initialized")
        except Exception as e:
            log.error(f"Failed to initialize YOLO detector: {e}")
            self.yolo_detector = None

    def signal_handler(self, sig, frame):
        """Handle system signals for clean exit"""
        log.info("Signal received, exiting...")
        State.running = False
        self.cleanup()
        sys.exit(0)

    def wait_for_game_window(self):
        """Wait for the game window to become available"""
        frame = grab_screen()
        if frame is not None and set_windows_offset(frame):
            return True
        return False

    def setup_visualization(self):
        """Set up visualization windows"""
        try:
            cv2.namedWindow("AutoDrive Visualization", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AutoDrive Visualization", 800, 600)
            cv2.namedWindow("AutoDrive Status", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AutoDrive Status", 400, 200)
            log.info("Visualization windows created")
        except Exception as e:
            log.error(f"Failed to create visualization windows: {e}")

    def update_status_display(self, frame):
        """Update the status display with current autodrive information"""
        try:
            status_frame = np.zeros((200, 400, 3), dtype=np.uint8)

            # Display autodrive status
            cv2.putText(
                status_frame,
                f"AutoDrive: {'ON' if State.autodrive_active else 'OFF'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if State.autodrive_active else (0, 0, 255), 2
            )

            # Display lane detection information if available
            if self.road_detector and hasattr(self.road_detector,
                                              'is_lane_detected') and self.road_detector.is_lane_detected():
                cv2.putText(status_frame, "Lane Detected: YES", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if hasattr(self.road_detector, 'get_drivable_direction'):
                    cv2.putText(status_frame,
                                f"Steering: {self.road_detector.get_drivable_direction():.1f} deg",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if hasattr(self.road_detector, 'get_recommended_speed'):
                    cv2.putText(status_frame,
                                f"Speed: {self.road_detector.get_recommended_speed():.2f}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(status_frame, "Lane Detected: NO", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display FPS information
            if self.frame_count > 0:
                fps = self.frame_count / (time.time() - self.start_time)
                cv2.putText(status_frame, f"FPS: {fps:.1f}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

            # Instructions
            cv2.putText(status_frame, "Z: Toggle AutoDrive", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("AutoDrive Status", status_frame)
        except Exception as e:
            log.error(f"Error updating status display: {e}")

    def run(self):
        """Main autodrive loop"""
        self.init_detectors()
        self.setup_visualization()

        log.info("Press Z to toggle AutoDrive, X for emergency stop, ESC to exit")

        # Main loop
        while State.running:
            # Update frame counter
            self.frame_count += 1
            current_time = time.time()

            # Print FPS every 5 seconds
            if current_time - self.last_fps_print >= 5.0:
                fps = self.frame_count / (current_time - self.start_time)
                log.info(f"FPS: {fps:.2f}")
                if fps < 10:
                    log.warning("Low FPS detected - may affect autodrive performance")
                self.last_fps_print = current_time

            # Rate limiting for performance
            if current_time - State.last_frame_time < 1.0 / State.target_fps:
                time.sleep(0.01)  # Short sleep to reduce CPU usage
                continue

            State.last_frame_time = current_time

            # Capture screen
            frame = grab_screen()
            if frame is None or not self.wait_for_game_window():
                time.sleep(0.5)
                continue

            # Process road detection
            lane_mask = None
            if self.road_detector:
                try:
                    self.road_detector.process_data()
                    lane_mask = getattr(self.road_detector, 'lane_mask', None)
                except Exception as e:
                    log.error(f"Error processing road data: {e}")

            # Run object detection
            detected_objects = []
            if self.yolo_detector:
                try:
                    results = self.yolo_detector.detect(frame)
                    detected_objects, _ = self.yolo_detector.process_detections(frame, results)
                except Exception as e:
                    log.error(f"Error in object detection: {e}")

            # Update visualization
            try:
                visualize_detections_and_lane(log, self.road_detector, frame, detected_objects, lane_mask)
            except Exception as e:
                log.error(f"Error in visualization: {e}")

            # Update status display
            self.update_status_display(frame)

            # Apply truck control if autodrive is active
            if State.autodrive_active:
                self.truck_controller.control(self.road_detector, detected_objects)

            # Check for window close
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                State.running = False
                break

        self.cleanup()

    def cleanup(self):
        """Clean up resources before exit"""
        log.info("Cleaning up resources...")
        cv2.destroyAllWindows()

        # Release all keys
        if hasattr(self, 'key_controller'):
            for key in ['w', 'a', 's', 'd']:
                self.key_controller.controller.release(key)

        log.info("AutoDrive terminated")


# Main entry point
if __name__ == '__main__':
    app = AutoDrive()
    app.run()