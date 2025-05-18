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
            return None, 0

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
            return None, 0

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

        # 计算偏移量的绝对值，用于动态调整转向时间
        error_magnitude = min(abs(error) / 100, 0.08)  # 最大转向时间为0.08秒

        # Determine steering action
        if abs(error) < self.lane_center_threshold:
            # Within threshold, no steering needed
            return None, 0
        elif steering > 0:
            return 'd', error_magnitude  # Turn right with dynamic duration
        else:
            return 'a', error_magnitude  # Turn left with dynamic duration

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

        # 计算总命令时长
        total_duration = 0.0
        max_duration = 0.1  # 最大总时长限制为0.1秒

        # 2. 计算转向动作和时长
        steering_action, steering_duration = self.calculate_steering(lane_data)

        if steering_action:
            # 添加转向命令，使用动态计算的持续时间
            commands.append([steering_action, f'{steering_duration:.2f}'])
            total_duration += steering_duration

        # 剩余时间用于加速
        remaining_duration = max(0.02, min(max_duration - total_duration, 0.06))

        # 3. 确保前进动作
        speed = status.get("speed", 0)
        gear = status.get("gear", "N")

        # 根据速度调整加速强度
        if speed < 5:
            # 低速时加大加速度
            commands.append(['w', f'{remaining_duration:.2f}'])
        else:
            # 高速时使用较小的加速度维持
            commands.append(['w', f'{remaining_duration * 0.7:.2f}'])

        # 处理交通灯
        if status.get("traffic_light") and len(status["traffic_light"]) > 0:
            traffic_cmd = self.process_traffic_light(status["traffic_light"])
            if traffic_cmd:
                # 如果需要处理交通灯，替换或添加对应命令
                commands = [traffic_cmd]

        self.last_taken_frame = status.get("detect_frame", None)
        # self.debug_command_save(commands, status.get("detect_frame", None))
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
                return ['s', '0.03']  # 减速并间隔0.03秒，减少黄灯时的减速时间

        return None  # 默认不操作

    def debug_command_save(self, command, frame):
        with open("debug_images/debug_commands.txt", "a") as f:
            f.write(f"Command: {command}, Frame: {frame}\n")