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

    ### modification: implement PID controller for lane following
    def calculate_steering(self, lane_data):
        """PID controller for lane centering"""
        if lane_data is None or not isinstance(lane_data, np.ndarray):
            return None

        # Calculate lane center and error
        # Assuming lane_data provides information about lane position
        h, w = lane_data.shape[:2] if len(lane_data.shape) > 2 else lane_data.shape

        # Define region of interest - look ahead for smoother steering
        look_ahead_y = int(h * 0.6)  # Look at 60% up the image for better anticipation

        # Find lane markings in the ROI
        row = lane_data[look_ahead_y, :]
        lane_pixels = np.where(row > 0)[0]

        if len(lane_pixels) < 2:
            # Not enough lane pixels detected
            return None

        # Calculate lane center and vehicle position
        lane_center = np.mean(lane_pixels)
        vehicle_center = w // 2

        # Calculate error (positive: need to turn right, negative: need to turn left)
        error = lane_center - vehicle_center

        # PID parameters - tune these values
        kp = 0.01  # Proportional gain
        ki = 0.001  # Integral gain
        kd = 0.005  # Derivative gain

        # Calculate PID terms
        p_term = kp * error
        self.error_sum += error
        i_term = ki * self.error_sum
        d_term = kd * (error - self.prev_error)
        self.prev_error = error

        # Calculate steering command
        steering = p_term + i_term + d_term

        # Determine steering action
        if abs(error) < self.lane_center_threshold:
            # Within threshold, no steering needed
            return None
        elif steering > 0:
            return 'd'  # Turn right
        else:
            return 'a'  # Turn left

    def get_action(self, status):
        """Determine driving actions based on lane data and vehicle state"""
        commands = []
        if self.last_taken_frame == status.get("detect_frame", None) and status.get("detect_frame", None) is not None:
            # Avoid repeated actions for the same frame
            return commands

        # Extract lane data safely
        ### modification: fixed the lane_status check to avoid array truth value error
        lane_data = None
        if status.get("lane_status") is not None:
            # Check if lane_status is a dictionary with lane_data key
            if isinstance(status["lane_status"], dict) and "lane_data" in status["lane_status"]:
                lane_data = status["lane_status"]["lane_data"]
            # Or if lane_data is directly in the output of detector
            elif "lane_data" in status:
                lane_data = status["lane_data"]
            # Or if lane_status directly contains lane_data (from LaneDetector)
            elif hasattr(status["lane_status"], "get") and status["lane_status"].get("lane_data") is not None:
                lane_data = status["lane_status"].get("lane_data")
            # If lane_status itself is the data
            elif isinstance(status["lane_status"], np.ndarray):
                lane_data = status["lane_status"]

        # If no valid lane data found, perform basic straight driving
        if lane_data is None:
            self.last_taken_frame = status.get("detect_frame", None)
            # No lane data available, maintain current speed but don't steer
            commands.append(['w', '0.2'])
            return commands

        speed = status.get("speed", 0)
        gear = status.get("gear", "N")

        ### modification: implement basic driving logic

        # 1. Handle initial acceleration from stop
        if speed < 5:
            if gear == "N" or gear == "A":
                # Start moving or keep moving forward slowly
                commands.append(['w', '0.5'])
            # Don't press 's' when stopped to avoid reversing

        # 2. Calculate steering action
        steering_action = self.calculate_steering(lane_data)
        if steering_action:
            # Add steering command with appropriate duration
            # Shorter duration for smoother control
            commands.append([steering_action, '0.2'])

        # 3. Maintain forward motion with controlled acceleration
        # Avoid sudden acceleration changes
        commands.append(['w', '0.2'])

        ### modification: Add obstacle avoidance (commented for future implementation)
        # if status.get("car_detect") and len(status["car_detect"]) > 0:
        #     # Handle obstacle detection and braking
        #     # Calculate distance and apply brakes if needed
        #     # commands.append(['s', '0.05'])
        #     pass
        self.last_taken_frame = status.get("detect_frame", None)

        return commands

    def process_traffic_light(self, traffic_light_data):
        """处理YOLO检测到的红绿灯数据，返回是否需要停车"""
        # 如果data数组中有Red的字符串则返回ImmediateStop，有Green则返回启动，Yellow则减速
        for data in traffic_light_data:
            if "Red" in data:
                return ['s', '0.05']  # 停车0.05秒
            elif "Green" in data:
                return ['w', '0.05']
            elif "Yellow" in data:
                return ['s', '0.1']  # 减速并间隔0.1秒

        return None  # 默认不操作