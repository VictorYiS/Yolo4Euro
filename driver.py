from log import log
import numpy as np


class TruckController():
    def __init__(self):
        self.autodrive_active = False
        self.prev_error = 0
        self.error_sum = 0
        self.lane_center_threshold = 20
        self.last_action_time = 0
        self.action_cooldown = 0.05
        self.prev_actions = []
        self.last_taken_frame = None
        # Add previous lane data storage
        self.prev_lane_data = None
        self.prev_lane_confidence = 0.0
        # Target right lane
        self.target_lane = 1  # 0-indexed, 1 = right lane

    def get_drive_status(self):
        return self.autodrive_active

    def drive_mode_toggle(self):
        self.autodrive_active = not self.autodrive_active
        ### modification: reset controllers when toggling mode
        self.prev_error = 0
        self.error_sum = 0
        self.prev_actions = []

    def calculate_steering(self, lane_data):
        """PID controller for right lane following with memory"""
        if lane_data is None and self.prev_lane_data is None:
            return None, 0

        # Use previous lane data if current is None or has few points
        current_data_valid = lane_data is not None and isinstance(lane_data, np.ndarray)
        if not current_data_valid and self.prev_lane_data is not None:
            # Decay confidence in old data
            self.prev_lane_confidence *= 0.8
            if self.prev_lane_confidence < 0.3:
                return None, 0  # Too uncertain to use old data
            lane_data = self.prev_lane_data
        elif current_data_valid:
            # Store current data for future use
            self.prev_lane_data = lane_data.copy()
            self.prev_lane_confidence = 1.0

        h, w = lane_data.shape[:2] if len(lane_data.shape) > 2 else lane_data.shape

        # Look at multiple heights for stability
        look_ahead_positions = [int(h * 0.6), int(h * 0.7)]
        all_lane_centers = []

        for y_pos in look_ahead_positions:
            if y_pos >= h:
                continue

            row = lane_data[y_pos, :]
            lane_pixels = np.where(row > 0)[0]

            if len(lane_pixels) < 5:  # Too few pixels
                continue

            # Group lane pixels into potential lane lines
            groups = []
            current_group = [lane_pixels[0]]

            for i in range(1, len(lane_pixels)):
                if lane_pixels[i] - lane_pixels[i - 1] < 20:  # Same line if close enough
                    current_group.append(lane_pixels[i])
                else:
                    if len(current_group) >= 3:  # Valid line needs minimum points
                        groups.append(current_group)
                    current_group = [lane_pixels[i]]

            if len(current_group) >= 3:
                groups.append(current_group)

            # Need at least 2 lane lines
            if len(groups) < 2:
                continue

            # Sort groups from left to right
            groups.sort(key=lambda g: np.mean(g))

            # Calculate lane centers (space between lines)
            lane_centers = []
            for i in range(len(groups) - 1):
                center = (np.mean(groups[i]) + np.mean(groups[i + 1])) / 2
                lane_centers.append(center)

            # If we have lane centers, record them
            if lane_centers:
                all_lane_centers.extend(lane_centers)

        # No valid lane centers found
        if not all_lane_centers:
            # Use previous error with decay for smooth transition
            reduced_steering = self.prev_error * 0.7
            if abs(reduced_steering) < self.lane_center_threshold:
                return None, 0
            elif reduced_steering > 0:
                return 'd', min(abs(reduced_steering) / 150, 0.05)
            else:
                return 'a', min(abs(reduced_steering) / 150, 0.05)

        # Find centers sorted left to right
        all_lane_centers.sort()

        # If we have enough lanes, target the right lane (index 1 for a 2-lane road)
        # Otherwise use rightmost detected lane
        target_idx = min(self.target_lane, len(all_lane_centers) - 1)
        target_center = all_lane_centers[target_idx]

        # Calculate error (distance from target lane center)
        vehicle_center = w // 2
        error = target_center - vehicle_center

        # Apply smoothing with previous error
        alpha = 0.7
        error = alpha * error + (1 - alpha) * self.prev_error

        # PID control
        kp = 0.01
        ki = 0.0005
        kd = 0.008

        # Anti-windup for integral term
        max_i_term = 10
        self.error_sum = max(min(self.error_sum + error, max_i_term), -max_i_term)

        p_term = kp * error
        i_term = ki * self.error_sum
        d_term = kd * (error - self.prev_error)
        self.prev_error = error

        steering = p_term + i_term + d_term

        # Dynamic steering time based on error magnitude
        error_magnitude = min(abs(error) / 150, 0.08)  # Softer steering

        if abs(error) < self.lane_center_threshold:
            return None, 0
        elif steering > 0:
            return 'd', error_magnitude
        else:
            return 'a', error_magnitude

    def get_action(self, status):
        """Determine driving actions with simultaneous key support"""
        if self.last_taken_frame == status.get("detect_frame", None) and status.get("detect_frame", None) is not None:
            return []

        # Extract lane data safely
        lane_data = self.extract_lane_data(status)

        # Check for traffic lights first as they have priority
        if status.get("traffic_light") and len(status["traffic_light"]) > 0:
            traffic_cmd = self.process_traffic_light(status["traffic_light"])
            if traffic_cmd:
                # Format: [['direction:duration', 'movement:duration']]
                return [[f'none:{traffic_cmd[1]}', f'{traffic_cmd[0]}:{traffic_cmd[1]}']]

        # Calculate steering with memory of previous lane data
        steering_action, steering_duration = self.calculate_steering(lane_data)

        # Implement action smoothing
        if steering_action:
            # Keep history of recent actions
            self.prev_actions.append((steering_action, steering_duration))
            if len(self.prev_actions) > 3:
                self.prev_actions.pop(0)

            # Smooth duration if we have history
            if len(self.prev_actions) >= 2:
                # Average recent durations, weighted toward current
                total_duration = sum(d for _, d in self.prev_actions[-2:])
                steering_duration = total_duration / 2

        # Calculate acceleration based on speed
        speed = status.get("speed", 0)

        # Adjust acceleration duration based on speed
        if speed < 5:
            accel_duration = max(0.02, min(0.1, 0.06))
        else:
            # Reduce acceleration at higher speeds
            accel_factor = max(0.5, min(1.0, 15.0 / speed))
            accel_duration = max(0.02, min(0.1, 0.06 * accel_factor))

        # Format for simultaneous key presses
        if steering_action:
            # Both steering and acceleration
            action = [[f'{steering_action}:{steering_duration:.2f}', f'w:{accel_duration:.2f}']]
        else:
            # Only acceleration, no steering
            action = [[f'none:{accel_duration:.2f}', f'w:{accel_duration:.2f}']]

        self.last_taken_frame = status.get("detect_frame", None)
        return action

    def extract_lane_data(self, status):
        """More robust extraction of lane data with multiple fallbacks"""
        lane_data = None
        speed_data = status.get("speed_data", None)
        set_speed_data = status.get("set_speed_data", None)

        # Try all possible ways to extract lane data
        if status.get("lane_status") is not None:
            if isinstance(status["lane_status"], dict) and "lane_data" in status["lane_status"]:
                lane_data = status["lane_status"]["lane_data"]
            elif isinstance(status["lane_status"], np.ndarray):
                lane_data = status["lane_status"]
            elif hasattr(status["lane_status"], "get"):
                lane_data = status["lane_status"].get("lane_data")

        if lane_data is None and "lane_data" in status:
            lane_data = status["lane_data"]

        if speed_data is not None:
            log.debug(f"Speed data: {speed_data}")
        if set_speed_data is not None:
            log.debug(f"Set speed data: {set_speed_data}")

        return lane_data

    def assess_lane_data_quality(self, lane_data):
        """Assess the quality/completeness of lane data"""
        if lane_data is None or not isinstance(lane_data, np.ndarray):
            return 0.0

        h, w = lane_data.shape[:2] if len(lane_data.shape) > 2 else lane_data.shape

        # Check multiple scan lines to assess overall quality
        quality_scores = []
        scan_heights = [int(h * 0.6), int(h * 0.7), int(h * 0.8)]

        for y_pos in scan_heights:
            if y_pos >= h:
                continue

            row = lane_data[y_pos, :]
            lane_pixels = np.where(row > 0)[0]

            # Score based on number of detected lane pixels
            if len(lane_pixels) == 0:
                quality_scores.append(0.0)
            else:
                # Higher score for more pixels, max at about 30 pixels
                score = min(1.0, len(lane_pixels) / 30.0)
                quality_scores.append(score)

        # Overall quality is average of scan line qualities
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    def process_traffic_light(self, traffic_light_data):
        """Process YOLO-detected traffic light data and return action"""
        for data in traffic_light_data:
            if "Red" in data:
                return ['s', '0.05']  # Stop for 0.05 seconds
            elif "Green" in data:
                return ['w', '0.05']  # Proceed for 0.05 seconds
            elif "Yellow" in data:
                return ['s', '0.03']  # Slow down for 0.03 seconds

        return None  # No action by default

    def debug_command_save(self, command, frame):
        with open("debug_images/debug_commands.txt", "a") as f:
            f.write(f"Command: {command}, Frame: {frame}\n")