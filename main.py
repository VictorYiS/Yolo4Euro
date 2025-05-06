# autodrive.py

import keyboard
import time
import sys
import signal
import cv2
import numpy as np
from grabscreen import grab_screen, init_camera
from window import BaseWindow, set_windows_offset, game_width, game_height
import logging
from pynput.keyboard import Key, Controller

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='autodrive.log',
    filemode='w'
)
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


class RoadDetector(BaseWindow):
    def __init__(self, sx, sy, ex, ey):
        super().__init__(sx, sy, ex, ey)
        self.lane_center = 0.0  # Position of lane center relative to screen center
        self.road_visible = False
        self.obstacle_detected = False
        self.distance_to_obstacle = 100.0  # meters

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

            log.debug(f"Lane center: {self.lane_center:.2f}, Road visible: {self.road_visible}, "
                      f"Obstacle: {self.obstacle_detected}, Distance: {self.distance_to_obstacle:.1f}m")

        except Exception as e:
            log.error(f"Error processing road data: {e}")


# Initialize screen regions
# These coordinates would need to be adjusted for EuroTruck Simulator 2
dashboard = TruckDashboard(100, 500, 400, 600)  # Example coordinates
road_view = RoadDetector(0, 0, game_width, game_height)


def signal_handler(sig, frame):
    global running
    log.info("Exiting...")
    running = False
    sys.exit(0)


def wait_for_game_window():
    """Wait until game window is detected"""
    log.info("Waiting for EuroTruck Simulator 2 window...")
    while running:
        frame = grab_screen()
        if frame is not None and set_windows_offset(frame):
            log.info("Game window detected!")
            return True
        time.sleep(1)
    return False


def press_key(key, duration=0.1):
    """Press and hold a key for a duration"""
    global last_key_press_time

    # Rate limiting to avoid flooding the game with inputs
    current_time = time.time()
    if current_time - last_key_press_time < key_press_interval:
        return

    last_key_press_time = current_time

    log.debug(f"Pressing key: {key} for {duration}s")
    kb_controller.press(key)
    time.sleep(duration)
    kb_controller.release(key)


def control_truck():
    """Apply control inputs based on detected road conditions"""
    global truck_steering

    # Only control if autodrive is active
    if not autodrive_active:
        return

    try:
        # Simple lane following logic
        if road_view.road_visible:
            # Convert lane position to steering input
            target_steering = -road_view.lane_center  # Negative because we want to steer toward the center

            # Apply smoothing to steering
            steering_delta = target_steering - truck_steering
            truck_steering += steering_delta * 0.2  # Gradual steering adjustment

            # Apply steering based on calculated value
            if truck_steering > 0.2:
                press_key('d', min(0.1, abs(truck_steering) * 0.2))
            elif truck_steering < -0.2:
                press_key('a', min(0.1, abs(truck_steering) * 0.2))

            # Speed control based on obstacles
            if road_view.obstacle_detected:
                if road_view.distance_to_obstacle < 30:
                    # Brake if obstacle is close
                    press_key('s', 0.2)
                else:
                    # Slow down
                    time.sleep(0.1)
            else:
                # Accelerate if road is clear
                press_key('w', 0.2)
        else:
            # If road isn't visible, slow down
            press_key('s', 0.1)

    except Exception as e:
        log.error(f"Error in truck control: {e}")


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

    # Toggle autodrive with z
    if keyboard.is_pressed('z'):
        autodrive_active = not autodrive_active
        log.info(f"AutoDrive {'activated' if autodrive_active else 'deactivated'}")
        time.sleep(0.5)  # Debounce

    # Emergency stop with Escape
    if keyboard.is_pressed('x'):
        if autodrive_active:
            autodrive_active = False
            log.info("Emergency stop activated")
            # Apply brakes
            kb_controller.press('s')
            time.sleep(1.0)
            kb_controller.release('s')
            time.sleep(0.5)  # Debounce


def main():
    global running

    log.info("Starting EuroTruck Simulator 2 AutoDrive")

    # Register signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize screen capture
    init_camera(target_fps)

    # Wait for game window
    if not wait_for_game_window():
        log.error("Failed to detect game window.")
        return

    log.info("Press z to toggle AutoDrive, x to emergency stop")

    try:
        # Main loop
        while running:
            # Check for user commands
            check_keyboard_commands()

            # Update screen information
            update_screen_info()

            # Control the truck if autodrive is active
            control_truck()

            # Sleep to avoid excessive CPU usage
            time.sleep(0.01)

    except KeyboardInterrupt:
        log.info("Exiting due to keyboard interrupt")
    except Exception as e:
        log.error(f"Unexpected error: {e}")
    finally:
        # Release all keys to avoid stuck inputs
        keys = ['w', 'a', 's', 'd']
        for key in keys:
            kb_controller.release(key)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()