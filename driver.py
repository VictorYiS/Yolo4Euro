from log import log
import numpy as np


class TruckController():
    def __init__(self):
        self.autodrive_active = False
        ### modification: add control parameters for better stability
        self.prev_error = 0
        self.error_sum = 0
        self.lane_center_threshold = 20  # px threshold for lane centering
        self.last_action_time = 0
        self.action_cooldown = 0.05  # seconds between actions
        self.prev_actions = []  # store recent actions for smoothing
        self.last_taken_frame = None

    def get_drive_status(self):
        return self.autodrive_active

    def drive_mode_toggle(self):
        self.autodrive_active = not self.autodrive_active
        ### modification: reset controllers when toggling mode
        self.prev_error = 0
        self.error_sum = 0
        self.prev_actions = []

    def calculate_steering(self, lane_data):
        """Enhanced PID controller for lane centering with robustness for incomplete lane data"""
        if lane_data is None or not isinstance(lane_data, np.ndarray):
            return None, 0

        h, w = lane_data.shape[:2] if len(lane_data.shape) > 2 else lane_data.shape

        # Use multiple scan lines at different heights for redundancy
        scan_heights = [int(h * 0.6), int(h * 0.7), int(h * 0.8)]
        valid_centers = []

        for y_pos in scan_heights:
            if y_pos >= h:
                continue

            row = lane_data[y_pos, :]
            lane_pixels = np.where(row > 0)[0]

            if len(lane_pixels) >= 2:
                # Calculate a potential lane center
                valid_centers.append(np.mean(lane_pixels))

        # If we don't have any valid data across scan lines, use previous steering
        if not valid_centers:
            # Gradually reduce previous steering if no new data
            reduced_steering = self.prev_error * 0.8

            if abs(reduced_steering) < self.lane_center_threshold:
                return None, 0
            elif reduced_steering > 0:
                return 'd', min(abs(reduced_steering) / 100, 0.05)
            else:
                return 'a', min(abs(reduced_steering) / 100, 0.05)

        # Use median filter to reject outliers from multiple scan lines
        lane_center = np.median(valid_centers)
        vehicle_center = w // 2

        # Apply exponential moving average for smoother response
        alpha = 0.7  # Smoothing factor
        if hasattr(self, 'smoothed_error'):
            error = alpha * (lane_center - vehicle_center) + (1 - alpha) * self.smoothed_error
        else:
            error = lane_center - vehicle_center

        self.smoothed_error = error

        # PID parameters - tune these values
        kp = 0.01  # Proportional gain
        ki = 0.0005  # Reduced integral gain to prevent oscillation
        kd = 0.008  # Increased derivative gain for better damping

        # Anti-windup for integral term to prevent overshoot
        max_i_term = 10
        self.error_sum = max(min(self.error_sum + error, max_i_term), -max_i_term)

        # Calculate PID terms
        p_term = kp * error
        i_term = ki * self.error_sum
        d_term = kd * (error - self.prev_error)
        self.prev_error = error

        # Calculate steering command with smoother response
        steering = p_term + i_term + d_term

        # Log stability data for debugging
        # log(f"Error: {error:.2f}, P: {p_term:.4f}, I: {i_term:.4f}, D: {d_term:.4f}, Steering: {steering:.4f}")

        # Dynamic steering time based on error magnitude and speed
        error_magnitude = min(abs(error) / 100, 0.08)

        # Add hysteresis to prevent oscillation near threshold
        steering_threshold = self.lane_center_threshold
        if hasattr(self, 'last_steering_dir'):
            # Increase threshold slightly if we're returning to center to prevent jitter
            if (self.last_steering_dir > 0 and error < 0) or (self.last_steering_dir < 0 and error > 0):
                steering_threshold += 5

        if abs(error) < steering_threshold:
            self.last_steering_dir = 0
            return None, 0
        elif steering > 0:
            self.last_steering_dir = 1
            return 'd', error_magnitude
        else:
            self.last_steering_dir = -1
            return 'a', error_magnitude

    def get_action(self, status):
        """Determine driving actions based on lane data and vehicle state with improved stability"""
        commands = []
        current_frame = status.get("detect_frame", None)

        # Skip duplicate frame processing
        if self.last_taken_frame == current_frame and current_frame is not None:
            return commands

        # Extract lane data with fallback mechanisms
        lane_data = self.extract_lane_data(status)

        # Track data quality for adaptive control
        data_quality = self.assess_lane_data_quality(lane_data)

        # Adaptive control based on data quality
        steering_action, steering_duration = self.calculate_steering(lane_data)

        # Store actions for smoothing
        if steering_action:
            # Add to history for temporal smoothing
            self.prev_actions.append((steering_action, float(steering_duration)))
            if len(self.prev_actions) > 5:
                self.prev_actions.pop(0)

            # Apply temporal smoothing for smoother transitions
            if len(self.prev_actions) >= 3:
                # Recent actions have more weight
                weights = [0.1, 0.2, 0.3, 0.5, 0.7][-len(self.prev_actions):]
                weighted_duration = sum(w * d for w, (_, d) in zip(weights, self.prev_actions)) / sum(weights)
                steering_duration = min(weighted_duration, 0.08)  # Cap at 0.08s

            # Add steering command with smoothed duration
            commands.append([steering_action, f'{steering_duration:.2f}'])

        # Adaptive acceleration based on current conditions
        speed = status.get("speed", 0)
        remaining_duration = min(0.1 - (steering_duration if steering_action else 0), 0.06)

        # Reduce acceleration when lane data quality is poor
        if data_quality < 0.5:
            remaining_duration *= 0.7  # Slow down when uncertain

        # Adaptive acceleration command
        if speed < 5:
            commands.append(['w', f'{remaining_duration:.2f}'])
        else:
            acceleration_factor = max(0.5, min(1.0, 10.0 / speed))  # Reduce acceleration at higher speeds
            commands.append(['w', f'{remaining_duration * acceleration_factor:.2f}'])

        # Process traffic signals with priority
        if status.get("traffic_light") and len(status["traffic_light"]) > 0:
            traffic_cmd = self.process_traffic_light(status["traffic_light"])
            if traffic_cmd:
                # Traffic commands take priority
                commands = [traffic_cmd]

        self.last_taken_frame = current_frame
        return commands

    def extract_lane_data(self, status):
        """More robust extraction of lane data with multiple fallbacks"""
        lane_data = None

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
        """处理YOLO检测到的红绿灯数据，返回是否需要停车"""
        # 如果data数组中有Red的字符串则返回ImmediateStop，有Green则返回启动，Yellow则减速
        for data in traffic_light_data:
            if "Red" in data:
                return ['s', '0.05']  # 停车0.05秒
            elif "Green" in data:
                return ['w', '0.05']
            elif "Yellow" in data:
                return ['s', '0.03']  # 减速并间隔0.03秒，减少黄灯时的减速时间

        return None  # 默认不操作

    def debug_command_save(self, command, frame):
        with open("debug_images/debug_commands.txt", "a") as f:
            f.write(f"Command: {command}, Frame: {frame}\n")