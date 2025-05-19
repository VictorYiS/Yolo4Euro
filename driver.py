from log import log
import numpy as np
import pickle
from collections import deque

from log import log


class TruckController():
    def __init__(self):
        # Core state
        self.autodrive_active = False
        self.target_lane = 1  # Default target lane (0-indexed, 1 = center lane in 3-lane, or right lane in 2-lane)

        # PID control parameters
        self.prev_error = 0
        self.error_sum = 0
        self.lane_center_threshold = 15

        # Lane tracking
        self.lane_history = deque(maxlen=5)  # Store recent lane positions
        self.prev_lane_data = None
        self.lane_confidence = 0.8
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

        # Lane count estimation
        self.lane_count = 2  # Default assumption: 2 lanes
        self.last_lane_count = 2
        self.stability_factor = 0.85  # Weight factor for temporal stability

        # Camera calibration
        self.camera_offset = 10  # Pixels to adjust for camera positioning

        # Action tracking
        self.last_action_time = 0
        self.prev_actions = []
        self.last_taken_frame = None

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
        self.lane_history.clear()

    def calculate_steering(self, lane_data):
        """Enhanced PID controller with improved lane memory and lane count handling"""
        # Process and validate lane data
        processed_data, confidence = self._process_lane_data(lane_data)

        # If no valid lanes detected, use historical data
        if processed_data is None or not processed_data.get('lane_centers', []):
            return self._use_historical_steering()

        # Store processed data for future reference
        self._update_lane_history(processed_data, confidence)

        # Determine target lane position
        target_position = self._determine_target_lane_position(processed_data)

        # Apply PID control to calculate steering action
        speed = processed_data.get('speed', 30)  # Default to moderate speed if unknown
        return self._apply_adaptive_pid(target_position, speed)

    def _process_lane_data(self, lane_data):
        """Process raw lane data with enhanced filtering and validation"""
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
            confidence = self.prev_lane_confidence
        elif current_data_valid:
            # Store current data for future use
            self.prev_lane_data = lane_data.copy()
            self.prev_lane_confidence = 1.0
            confidence = 1.0
        else:
            return None, 0

        # Get image dimensions
        h, w = lane_data.shape[:2] if len(lane_data.shape) > 2 else lane_data.shape

        # Use multiple scan lines at different heights for more robust detection
        look_ahead_positions = [int(h * 0.4), int(h * 0.5), int(h * 0.6), int(h * 0.7)]
        all_lane_centers = []
        all_weights = []
        valid_lane_count = 0

        # Process each scan line
        for y_pos in look_ahead_positions:
            if y_pos >= h:
                continue

            # Extract and analyze row data
            detected_centers, is_valid = self._analyze_scan_line(lane_data, y_pos, w)

            if is_valid:
                valid_lane_count += 1

                # Weight by vertical position (closer = more important)
                weight = 1.0 - (y_pos / h)

                for center in detected_centers:
                    all_lane_centers.append(center)
                    all_weights.append(weight)

        # Update lane count estimate
        if valid_lane_count >= 2 and all_lane_centers:
            unique_centers = self._cluster_lane_centers(all_lane_centers)
            estimated_lane_count = len(unique_centers)

            # Apply temporal smoothing to lane count
            self.lane_count = int(self.stability_factor * self.last_lane_count +
                                  (1 - self.stability_factor) * estimated_lane_count)
            self.last_lane_count = self.lane_count

        # Return processed data and confidence
        return {
            'lane_centers': all_lane_centers,
            'weights': all_weights,
            'clustered_centers': self._cluster_lane_centers(all_lane_centers) if all_lane_centers else [],
            'lane_count': self.lane_count,
            'speed': 0  # Will be updated by caller if available
        }, confidence

    def _analyze_scan_line(self, lane_data, y_pos, image_width):
        """Analyze a horizontal scan line to detect lane positions"""
        # Extract row data
        row = lane_data[y_pos, :]
        lane_pixels = np.where(row > 0)[0]

        # Not enough pixels for valid analysis
        if len(lane_pixels) < 5:
            return [], False

        # Group lane pixels into potential lane lines
        # Larger gap threshold for further distances (wider lanes in perspective)
        gap_threshold = max(10, 20 - int((y_pos / lane_data.shape[0]) * 10))
        groups = self._group_lane_pixels(lane_pixels, gap_threshold)

        # Need at least 2 groups to form a lane
        if len(groups) < 2:
            return [], False

        # Calculate width of each potential lane
        lane_centers, lane_widths = self._calculate_lane_centers_and_widths(groups)

        # Validate lanes based on expected width
        valid_centers = self._validate_lanes(lane_centers, lane_widths)

        return valid_centers, len(valid_centers) > 0

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

    def _cluster_lane_centers(self, centers, threshold=30):
        """Cluster lane centers to identify unique lanes"""
        if not centers:
            return []

        # Sort centers
        centers = sorted(centers)
        clusters = [[centers[0]]]

        # Group close centers
        for center in centers[1:]:
            if center - clusters[-1][-1] < threshold:
                clusters[-1].append(center)
            else:
                clusters.append([center])

        # Calculate average center for each cluster
        return [sum(cluster) / len(cluster) for cluster in clusters]

    def _update_lane_history(self, processed_data, confidence):
        """Update lane history with current detection"""
        clustered_centers = processed_data.get('clustered_centers', [])

        if clustered_centers:
            # Update lane width estimate if we have multiple lanes
            if len(clustered_centers) >= 2:
                widths = []
                for i in range(len(clustered_centers) - 1):
                    width = clustered_centers[i + 1] - clustered_centers[i]
                    if 60 < width < 200:  # Reasonable lane width range
                        widths.append(width)

                if widths:
                    # Use median for robustness against outliers
                    median_width = sorted(widths)[len(widths) // 2]
                    # Adaptive learning rate based on confidence
                    alpha = self.lane_width_alpha * confidence
                    self.expected_lane_width = (1 - alpha) * self.expected_lane_width + alpha * median_width

            # Add to lane history
            self.lane_history.append(clustered_centers)
            self.lane_confidence = confidence

    def _determine_target_lane_position(self, processed_data):
        """Determine target lane position based on detected lanes and road configuration"""
        clustered_centers = processed_data.get('clustered_centers', [])

        if not clustered_centers:
            # No centers detected, fall back to previous error
            return self.prev_error

        # Sort centers from left to right
        sorted_centers = sorted(clustered_centers)

        # Determine target lane based on estimated lane count
        lane_count = processed_data.get('lane_count', 2)

        if lane_count >= 3:
            # On 3+ lane roads, target center lane
            if len(sorted_centers) >= 3:
                # Can identify center lane
                middle_index = len(sorted_centers) // 2
                target = sorted_centers[middle_index]
            elif len(sorted_centers) == 2:
                # Estimate center between detected lanes
                target = (sorted_centers[0] + sorted_centers[1]) / 2
            else:
                # Only one lane, use it
                target = sorted_centers[0]
        else:
            # On 2-lane roads, target right lane
            if len(sorted_centers) >= 2:
                target = sorted_centers[-2]  # Second-to-last lane (right lane)
            else:
                target = sorted_centers[-1]  # Rightmost detected lane

        # Apply temporal smoothing for stability
        if len(self.lane_history) > 0:
            # Use weighted moving average with more weight to recent values
            weights = [0.6, 0.75, 0.85, 0.95, 1.0]
            weights = weights[-len(self.lane_history):]

            # Calculate image center with camera offset
            image_width = 640  # Default width, should be updated in production
            if self.prev_lane_data is not None:
                h, image_width = self.prev_lane_data.shape[:2] if len(
                    self.prev_lane_data.shape) > 2 else self.prev_lane_data.shape
            vehicle_center = image_width // 2 + self.camera_offset

            # Calculate error (difference from vehicle center)
            return target - vehicle_center

        return target

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

    def _apply_adaptive_pid(self, error, speed):
        """Apply PID control with parameters that adapt to driving conditions"""
        # Base PID parameters
        kp_base = 0.01
        ki_base = 0.0005
        kd_base = 0.008

        # Adapt parameters based on speed
        speed_factor = min(1.0, max(0.5, speed / 30.0)) if speed > 0 else 0.5
        kp = kp_base * (1 + 0.2 * (1 - speed_factor))  # More aggressive at lower speeds
        ki = ki_base * speed_factor  # Less integral at high speeds
        kd = kd_base * (1 + 0.5 * speed_factor)  # More dampening at higher speeds

        # Smoother error with adaptive filtering
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
        """Determine driving actions with enhanced decision-making and smoother control"""
        # Avoid processing the same frame twice
        if self.last_taken_frame == status.get("detect_frame", None) and status.get("detect_frame", None) is not None:
            return []

        # Extract lane data and other status information
        lane_data = self._extract_lane_data(status)
        speed = status.get("speed", 0)

        # Process vehicle detection data
        car_list = self._process_car_detection(status)

        # Process vehicle detection and lane categorization
        lane_status = status.get("lane_status") or {}
        cars_in_lanes = []
        if car_list and isinstance(lane_status, dict):
            cars_in_lanes = self.categorize_cars_by_lane(lane_status, car_list)
            for obj, side in cars_in_lanes:
                log.debug(f"Vehicle: {obj['class_name']} in lane: {side}")

        # Check for traffic lights first (highest priority)
        if status.get("traffic_light") and len(status["traffic_light"]) > 0:
            traffic_cmd = self._process_traffic_light(status["traffic_light"])
            if traffic_cmd:
                self.last_taken_frame = status.get("detect_frame", None)
                return [[f'none:{traffic_cmd[1]}', f'{traffic_cmd[0]}:{traffic_cmd[1]}']]

        # Detect road conditions (curves, etc.)
        road_conditions = self._detect_road_conditions(lane_data, cars_in_lanes)

        # Calculate steering action
        steering_action, steering_duration = self.calculate_steering(lane_data)

        # Calculate acceleration based on speed and road conditions
        accel_action, accel_duration = self._calculate_acceleration(speed, road_conditions)

        # Apply action smoothing
        action = self._create_combined_action(steering_action, steering_duration,
                                              accel_action, accel_duration)

        self.last_taken_frame = status.get("detect_frame", None)
        return action

    def _extract_lane_data(self, status):
        """Extract lane data from status with multiple fallbacks"""
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

    def _process_car_detection(self, status):
        """Process car detection data safely"""
        car_raw = status.get("car_detect")

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

        return car_list

    def _detect_road_conditions(self, lane_data, cars_in_lanes):
        """Detect road conditions like curves, obstacles, etc."""
        conditions = {
            'is_curve': False,
            'has_obstacle': False,
            'obstacle_side': None,
            'recommend_lane_change': False
        }

        # Detect curves from lane data
        if lane_data is not None and isinstance(lane_data, np.ndarray):
            try:
                # Check if lanes have significant curvature
                h, w = lane_data.shape[:2] if len(lane_data.shape) > 2 else lane_data.shape

                # Analyze several horizontal scan lines
                upper_scan = int(h * 0.4)
                lower_scan = int(h * 0.7)

                if upper_scan < h and lower_scan < h:
                    upper_row = lane_data[upper_scan, :]
                    lower_row = lane_data[lower_scan, :]

                    upper_lane_pixels = np.where(upper_row > 0)[0]
                    lower_lane_pixels = np.where(lower_row > 0)[0]

                    if len(upper_lane_pixels) > 0 and len(lower_lane_pixels) > 0:
                        # Calculate lane center shift between upper and lower scan lines
                        upper_center = np.mean(upper_lane_pixels)
                        lower_center = np.mean(lower_lane_pixels)

                        # If significant shift, it's likely a curve
                        if abs(upper_center - lower_center) > w * 0.1:
                            conditions['is_curve'] = True
            except Exception as e:
                log.error(f"Error detecting road curve: {e}")

        # Detect obstacles from cars_in_lanes
        if cars_in_lanes:
            for car, lane in cars_in_lanes:
                # If car is in our lane and close
                if lane == "right" and self.lane_count <= 2:
                    # Calculate distance/size ratio as proxy for proximity
                    x1, y1, x2, y2 = car['bbox']
                    car_height = y2 - y1

                    # If car is tall in image (close to us)
                    if car_height > 100:  # Adjust threshold as needed
                        conditions['has_obstacle'] = True
                        conditions['obstacle_side'] = "front"

                        # If left lane is clear, recommend lane change
                        if not any(l == "left" for _, l in cars_in_lanes):
                            conditions['recommend_lane_change'] = True

        return conditions

    def _calculate_acceleration(self, speed, road_conditions):
        """Calculate acceleration action based on speed and road conditions"""
        # Base acceleration parameters
        if speed < 5:
            # Higher acceleration at low speeds
            accel_duration = self.base_accel_duration * 1.2
        else:
            # Reduce acceleration at higher speeds for stability
            accel_factor = max(0.6, min(1.0, 15.0 / speed)) if speed > 0 else 1.0
            accel_duration = self.base_accel_duration * accel_factor

        # Adjust for road conditions
        if road_conditions.get('is_curve', False):
            # Reduce speed in curves
            if speed > 60:
                return 's', 0.03  # Light braking
            elif speed < 40:
                return 'w', accel_duration * 0.8  # Gentle acceleration
            else:
                return 'w', accel_duration * 0.5  # Maintain speed

        if road_conditions.get('has_obstacle', False):
            # Adjust for obstacles
            if road_conditions.get('obstacle_side') == "front":
                if speed > 30:
                    return 's', 0.04  # Moderate braking
                else:
                    return 'none', 0  # Coast

        # Normal driving - constrain within limits
        accel_duration = max(self.min_accel_duration, min(self.max_accel_duration, accel_duration))
        return 'w', accel_duration

    def _create_combined_action(self, steering_action, steering_duration, accel_action, accel_duration):
        """Create combined steering and acceleration action"""
        # Record action for history
        if steering_action:
            # Track steering actions for smoothing
            self.prev_actions.append((steering_action, steering_duration))
            if len(self.prev_actions) > 3:
                self.prev_actions.pop(0)

            # Smooth duration using recent history
            if len(self.prev_actions) >= 2:
                # Average recent durations, weighted toward current
                total_duration = sum(d for _, d in self.prev_actions[-2:])
                steering_duration = total_duration / 2

        # Format for simultaneous key presses
        if steering_action and accel_action != 'none':
            # Both steering and acceleration/braking
            action = [[f'{steering_action}:{steering_duration:.2f}', f'{accel_action}:{accel_duration:.2f}']]
        elif steering_action:
            # Only steering
            action = [[f'{steering_action}:{steering_duration:.2f}', f'none:{steering_duration:.2f}']]
        elif accel_action != 'none':
            # Only acceleration/braking
            action = [[f'none:{accel_duration:.2f}', f'{accel_action}:{accel_duration:.2f}']]
        else:
            # No action
            action = []

        return action

    def _process_traffic_light(self, traffic_light_data):
        """Process traffic light data and return appropriate action"""
        for data in traffic_light_data:
            if "Red" in data:
                return ['s', '0.05']  # Stop for 0.05 seconds
            elif "Green" in data:
                return ['w', '0.05']  # Proceed for 0.05 seconds
            elif "Yellow" in data:
                return ['s', '0.03']  # Slow down for 0.03 seconds

        return None  # No action by default