from log import log
import numpy as np
import pickle

class TruckController():
    def __init__(self):
        self.autodrive_active = False
        self.prev_error = 0
        self.error_sum = 0
        self.lane_center_threshold = 20
        self.last_action_time = 0
        self.prev_actions = []
        self.last_taken_frame = None

        # Enhanced lane memory
        self.prev_lane_data = None
        self.prev_lane_confidence = 0.0
        self.lane_history = []  # Store recent lane centers for stability
        self.max_history = 5

        # Target right lane by default (0-indexed, 1 = right lane)
        self.target_lane = 1

        # Camera offset compensation (slight right bias in camera)
        self.camera_offset = 10  # Pixels to adjust for camera positioning

        # Lane width estimation for validation
        self.expected_lane_width = 120  # Initial estimate, will adapt
        self.lane_width_alpha = 0.1  # Adaptation rate

        # Speed control parameters
        self.min_accel_duration = 0.02
        self.max_accel_duration = 0.08
        self.base_accel_duration = 0.06

    def get_drive_status(self):
        return self.autodrive_active

    def drive_mode_toggle(self):
        self.autodrive_active = not self.autodrive_active
        # Reset controllers when toggling mode
        self.prev_error = 0
        self.error_sum = 0
        self.prev_actions = []
        self.lane_history = []

    def calculate_steering(self, lane_data):
        """Enhanced PID controller for right lane following with improved memory and reliability"""
        # If no current or previous data, cannot steer
        if lane_data is None and self.prev_lane_data is None:
            return None, 0

        # Decide whether to use current or previous lane data
        current_data_valid = lane_data is not None and isinstance(lane_data, np.ndarray)

        # Use previous lane data if current is None or has few points
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

        # Multiple scan lines for better lane detection
        # Look further ahead for stability, closer for immediate corrections
        look_ahead_positions = [int(h * 0.5), int(h * 0.6), int(h * 0.7)]
        detected_lanes = []

        for y_pos in look_ahead_positions:
            if y_pos >= h:
                continue

            # Extract row data
            row = lane_data[y_pos, :]
            lane_pixels = np.where(row > 0)[0]

            if len(lane_pixels) < 5:  # Too few pixels
                continue

            # Group lane pixels into potential lane lines with dynamic gap threshold
            # Larger gap threshold for further distances (wider lanes in perspective)
            gap_threshold = max(10, 20 - int((y_pos / h) * 10))
            groups = self._group_lane_pixels(lane_pixels, gap_threshold)

            # Need at least 2 groups to form a lane
            if len(groups) < 2:
                continue

            # Calculate width of each potential lane
            lane_centers, lane_widths = self._calculate_lane_centers_and_widths(groups)

            # Validate lanes based on expected width
            valid_centers = self._validate_lanes(lane_centers, lane_widths)

            # Add to detected lanes with y-position as weight (closer = more important)
            weight = 1.0 - (y_pos / h)
            for center in valid_centers:
                detected_lanes.append((center, weight))

        # If no valid lanes detected, use history with decay
        if not detected_lanes:
            return self._use_historical_steering()

        # Calculate weighted average of lane centers
        all_lane_centers = [center for center, _ in detected_lanes]
        all_weights = [weight for _, weight in detected_lanes]

        # Sort centers from left to right
        sorted_centers = sorted(all_lane_centers)

        # Update lane width estimate if we have enough lanes
        if len(sorted_centers) >= 2:
            for i in range(len(sorted_centers) - 1):
                width = sorted_centers[i + 1] - sorted_centers[i]
                if 60 < width < 200:  # Reasonable lane width in pixels
                    self.expected_lane_width = (1 - self.lane_width_alpha) * self.expected_lane_width + \
                                               self.lane_width_alpha * width

        # Determine target lane center
        target_center = self._determine_target_lane_center(sorted_centers)

        # Add to lane history for stability
        self.lane_history.append(target_center)
        if len(self.lane_history) > self.max_history:
            self.lane_history.pop(0)

        # Use weighted moving average for target
        weights = [0.6, 0.75, 0.85, 0.95, 1.0]  # More weight to recent values
        weights = weights[-len(self.lane_history):]
        smoothed_target = sum(c * w for c, w in zip(self.lane_history, weights)) / sum(weights)

        # Calculate error with camera offset compensation
        vehicle_center = w // 2 + self.camera_offset
        error = smoothed_target - vehicle_center

        # Apply PID control
        steering_action, steering_duration = self._apply_pid_control(error)

        return steering_action, steering_duration

    def _group_lane_pixels(self, lane_pixels, gap_threshold=20):
        """Group lane pixels into potential lane lines"""
        groups = []
        current_group = [lane_pixels[0]]

        for i in range(1, len(lane_pixels)):
            if lane_pixels[i] - lane_pixels[i - 1] < gap_threshold:  # Same line if close enough
                current_group.append(lane_pixels[i])
            else:
                if len(current_group) >= 3:  # Valid line needs minimum points
                    groups.append(current_group)
                current_group = [lane_pixels[i]]

        if len(current_group) >= 3:
            groups.append(current_group)

        return groups

    def _calculate_lane_centers_and_widths(self, groups):
        """Calculate centers between lane lines and their widths"""
        # Sort groups from left to right
        groups.sort(key=lambda g: np.mean(g))

        # Calculate lane centers (space between lines)
        lane_centers = []
        lane_widths = []

        for i in range(len(groups) - 1):
            left_line = np.mean(groups[i])
            right_line = np.mean(groups[i + 1])
            center = (left_line + right_line) / 2
            width = right_line - left_line

            lane_centers.append(center)
            lane_widths.append(width)

        return lane_centers, lane_widths

    def _validate_lanes(self, centers, widths):
        """Validate lane centers based on expected width"""
        valid_centers = []

        # Accept lanes with reasonable width
        width_tolerance = 0.4  # 40% tolerance
        min_valid_width = self.expected_lane_width * (1 - width_tolerance)
        max_valid_width = self.expected_lane_width * (1 + width_tolerance)

        for i, (center, width) in enumerate(zip(centers, widths)):
            # Accept if width is reasonable or if we don't have many options
            if min_valid_width <= width <= max_valid_width or len(centers) <= 2:
                valid_centers.append(center)

        return valid_centers

    def _use_historical_steering(self):
        """Use historical steering data when no lanes detected"""
        # Use previous error with decay for smooth transition
        reduced_steering = self.prev_error * 0.7

        if abs(reduced_steering) < self.lane_center_threshold:
            return None, 0
        elif reduced_steering > 0:
            return 'd', min(abs(reduced_steering) / 150, 0.05)
        else:
            return 'a', min(abs(reduced_steering) / 150, 0.05)

    def _determine_target_lane_center(self, sorted_centers):
        """Determine which lane center to target"""
        # Target rightmost lane by default
        if not sorted_centers:
            # No centers detected, use previous error
            return self.prev_error

        # If we have at least two lanes, we can try to identify the right lane
        if len(sorted_centers) >= 2:
            # For a two-lane road with 3 lines, the right lane center is typically index 1
            # For roads with more lanes, we prioritize staying in a right lane, but not the shoulder

            # Check if we might have a shoulder/barrier detection (much wider than a normal lane)
            if len(sorted_centers) >= 3:
                right_width = sorted_centers[-1] - sorted_centers[-2]
                if right_width > self.expected_lane_width * 1.5:
                    # Likely a shoulder/barrier - use second-to-last center
                    return sorted_centers[-2]

            # Use second-from-right for safety (right lane but not shoulder)
            if len(sorted_centers) >= 2:
                return sorted_centers[-2]

        # Default to rightmost detected lane if nothing else works
        return sorted_centers[-1]

    def _apply_pid_control(self, error):
        """Apply PID control to calculate steering"""
        # Fine-tuned PID parameters
        kp = 0.01  # Proportional gain
        ki = 0.0005  # Integral gain
        kd = 0.008  # Derivative gain

        # Apply smoothing with previous error
        alpha = 0.7
        smoothed_error = alpha * error + (1 - alpha) * self.prev_error

        # Anti-windup for integral term
        max_i_term = 10
        self.error_sum = max(min(self.error_sum + smoothed_error, max_i_term), -max_i_term)

        # Calculate PID terms
        p_term = kp * smoothed_error
        i_term = ki * self.error_sum
        d_term = kd * (smoothed_error - self.prev_error)

        # Update previous error
        self.prev_error = smoothed_error

        # Calculate total steering
        steering = p_term + i_term + d_term

        # Dynamic steering time based on error magnitude with dampening
        # Lower value for gentler corrections
        error_magnitude = min(abs(smoothed_error) / 180, 0.07)

        # Different steering thresholds based on magnitude
        if abs(smoothed_error) < self.lane_center_threshold:
            return None, 0
        elif steering > 0:
            return 'd', error_magnitude
        else:
            return 'a', error_magnitude
    def categorize_cars_by_lane(self, lane_status, car_list):
        """
        给定 lane_status（含 'lane_data' 和 'instance_data' np.ndarray）
        以及车列表 car_list（每个 dict 有 'bbox'），
        返回一个列表，每项是 (obj, lane_side)：
            lane_side in {"left","right"} 或 None（不在任何车道）
        """
        lane_data = lane_status.get("lane_data", None)
        inst_data = lane_status.get("instance_data", None)
        if lane_data is None or inst_data is None:
            return [(obj, None) for obj in car_list]

        H, W = inst_data.shape[:2]

        # 1. 找到所有实例 ID（排除背景 0）
        ids = np.unique(inst_data)
        ids = ids[ids != 0]

        # 2. 计算每个实例在水平上的平均位置
        id_positions = []
        for inst_id in ids:
            ys, xs = np.where(inst_data == inst_id)
            if len(xs) == 0: 
                continue
            mean_x = xs.mean()
            id_positions.append((inst_id, mean_x))

        # 如果一条都没检测到，全部标 None
        if not id_positions:
            return [(obj, None) for obj in car_list]

        # 3. 按 mean_x 排序，从左到右
        id_positions.sort(key=lambda x: x[1])
        # 4. 映射到 “left”/“right”
        side_map = {}
        if len(id_positions) == 1:
            # 只有一条车道，我们当作 center，也可以标成 left
            side_map[id_positions[0][0]] = "center"
        else:
            # 最左的是 left，最右的是 right，若中间还能扩
            side_map[id_positions[0][0]] = "left"
            side_map[id_positions[-1][0]] = "right"
            # 中间的全标 middle
            for inst_id, _ in id_positions[1:-1]:
                side_map[inst_id] = "middle"

        # 5. 给每辆车做归属
        result = []
        for obj in car_list:
            x1, y1, x2, y2 = obj['bbox']
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            # 限定在图像范围内
            cx = np.clip(cx, 0, W-1)
            cy = np.clip(cy, 0, H-1)

            inst_id = int(inst_data[cy, cx])
            lane_side = side_map.get(inst_id, None)
            result.append((obj, lane_side))

        return result
    def get_action(self, status):
        """Determine driving actions with simultaneous key support"""
        # Avoid processing the same frame twice
        if self.last_taken_frame == status.get("detect_frame", None) and status.get("detect_frame", None) is not None:
            return []

        # Extract lane data safely
        lane_data = self.extract_lane_data(status)

#车辆检测，但是lane_status复用了，后期可以合并一下
        car_raw = status.get("car_detect")  
  
        # 根据类型决定如何处理

        if isinstance(car_raw, (bytes, bytearray)):
            # 真的是 pickle.dumps 过的字节流，就 loads
            try:
                car_list = pickle.loads(car_raw)
            except Exception as e:
                log.error(f"Failed to unpickle car_detect: {e}")
                car_list = []
        elif isinstance(car_raw, list):
            # 已经是 list 了，直接用
            car_list = car_raw
        else:
            # 其它情况都当成空列表
            car_list = []
        # 同时确保 lane_status 也合法
        lane_status = status.get("lane_status") or {}
        if car_list and isinstance(lane_status, dict):
            categorized = self.categorize_cars_by_lane(lane_status, car_list)
            for obj, side in categorized:
                left_cars   = [obj for obj, side in categorized if side == "left"]
                middle_cars = [obj for obj, side in categorized if side == "middle"]
                right_cars  = [obj for obj, side in categorized if side == "right"]

                print("******************")
                print(
                    f"[DEBUG] {obj['class_name']} bbox={obj['bbox']} → lane_side={side}"
                )


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

        # Ensure more consistent forward movement
        # Higher base acceleration at low speeds, reduced at higher speeds
        if speed < 5:
            accel_duration = self.base_accel_duration * 1.2
        else:
            # Gradually reduce acceleration at higher speeds for stability
            accel_factor = max(0.6, min(1.0, 15.0 / speed))
            accel_duration = self.base_accel_duration * accel_factor

        # Constrain within limits
        accel_duration = max(self.min_accel_duration, min(self.max_accel_duration, accel_duration))

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

        # Log speed data if available
        speed_data = status.get("speed_data", None)
        set_speed_data = status.get("set_speed_data", None)

        if speed_data is not None:
            log.debug(f"Speed data: {speed_data}")
        if set_speed_data is not None:
            log.debug(f"Set speed data: {set_speed_data}")

        return lane_data

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