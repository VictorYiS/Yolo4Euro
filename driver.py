import time

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
        self.max_history = 5

        # Lane count estimation
        self.lane_count = 2  # Default assumption: 2 lanes
        self.last_lane_count = 2
        self.stability_factor = 0.85  # Weight factor for temporal stability

        # Lane width estimation for validation
        self.expected_lane_width = 150  # Initial estimate, will adapt
        self.lane_width_alpha = 0.1  # Adaptation rate

        # Camera calibration
        self.camera_offset = 10  # Pixels to adjust for camera positioning

        # Curve detection
        self.is_curve_detected = False
        self.curve_direction = None
        self.previous_valid_centers = []

        # Action tracking
        self.last_action_time = 0
        self.prev_actions = []
        self.last_taken_frame = None

        # Speed control parameters
        self.min_accel_duration = 0.02
        self.max_accel_duration = 0.08
        self.base_accel_duration = 0.06

        # Constants (replacing magic numbers)
        self.LANE_PIXEL_MIN = 5
        self.LANE_GAP_BASE = 10
        self.CENTER_CONSISTENCY_THRESHOLD = 30
        self.CURVE_DETECTION_THRESHOLD = 0.2
        self.MIN_SCAN_LINES_FOR_CURVE = 3

    def get_drive_status(self):
        return self.autodrive_active

    def drive_mode_toggle(self):
        self.autodrive_active = not self.autodrive_active
        # Reset controllers when toggling mode
        self.prev_error = 0
        self.error_sum = 0
        self.prev_actions = []
        self.lane_history.clear()

    def calculate_steering(self, lane_data, game_steer=0.0):
        """Enhanced PID controller with improved lane memory and lane count handling,
        now considering current steering angle"""
        # Process and validate lane data
        processed_data, confidence = self._process_lane_data(lane_data)

        # If no valid lanes detected, use historical data
        if processed_data is None or not processed_data.get('lane_centers', []):
            return self._use_historical_steering(game_steer)

        # Store processed data for future reference
        self._update_lane_history(processed_data, confidence)

        # Determine target lane position
        target_position = self._determine_target_lane_position(processed_data)

        # Apply PID control to calculate steering action
        speed = processed_data.get('speed', 30)  # Default to moderate speed if unknown
        return self._apply_adaptive_pid(target_position, speed, game_steer)

    def _process_lane_data(self, lane_data):
        """Process raw lane data with enhanced filtering and validation - combines former
        _process_lane_data and _analyze_scan_line functions"""
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
        look_ahead_positions = [int(h * 0.3), int(h * 0.4), int(h * 0.5), int(h * 0.6), int(h * 0.7)]
        all_lane_centers = []
        all_weights = []
        valid_lane_count = 0
        scan_line_counts = []  # Track counts per scan line for detecting curves

        # Process each scan line
        for y_pos in look_ahead_positions:
            if y_pos >= h:
                continue

            # Extract row data for this scan line
            row = lane_data[y_pos, :]
            lane_pixels = np.where(row > 0)[0]

            # Not enough pixels for valid analysis
            if len(lane_pixels) < self.LANE_PIXEL_MIN:
                scan_line_counts.append(0)
                continue

            # Group lane pixels into potential lane lines
            # Larger gap threshold for further distances (wider lanes in perspective)
            gap_threshold = max(self.LANE_GAP_BASE, 20 - int((y_pos / h) * 10))
            groups = self._group_lane_pixels(lane_pixels, gap_threshold)

            # Need at least 2 groups to form a lane
            if len(groups) < 2:
                scan_line_counts.append(0)
                continue

            # Apply directional filters for curve detection
            groups = self._filter_groups_by_direction(groups, y_pos, lane_data)

            # Calculate centers between lane lines and their widths
            lane_centers, lane_widths = [], []
            if len(groups) >= 2:
                # Sort groups from left to right
                groups.sort(key=lambda g: np.mean(g))

                # Calculate lane centers (space between lines)
                for i in range(len(groups) - 1):
                    left_line = np.mean(groups[i])
                    right_line = np.mean(groups[i + 1])
                    center = (left_line + right_line) / 2
                    width = right_line - left_line

                    lane_centers.append(center)
                    lane_widths.append(width)

            # Validate lanes based on expected width and curve-aware validation
            valid_centers = []
            if lane_centers:
                # Adjust tolerance based on curve detection and scan line position
                is_upper_scan = y_pos < (h * 0.5)

                # Determine appropriate width tolerance
                base_tolerance = 0.4  # 40% base tolerance
                width_tolerance = (base_tolerance + 0.2 if self.is_curve_detected else
                                   (base_tolerance + 0.1 if is_upper_scan else base_tolerance))

                min_valid_width = self.expected_lane_width * (1 - width_tolerance)
                max_valid_width = self.expected_lane_width * (1 + width_tolerance)

                for i, (center, width) in enumerate(zip(lane_centers, lane_widths)):
                    # Base validation on width constraints
                    width_valid = min_valid_width <= width <= max_valid_width

                    # Check for consistency with previous centers
                    center_valid = len(lane_centers) <= 2  # Always accept if only a few lanes

                    if not width_valid and self.previous_valid_centers:
                        # Check if this center is close to any previously valid center
                        center_valid = any(abs(center - prev_center) < self.CENTER_CONSISTENCY_THRESHOLD
                                           for prev_center in self.previous_valid_centers)

                    # Accept if either width is reasonable or center is consistent with history
                    if width_valid or (center_valid and len(lane_centers) <= 3):
                        valid_centers.append(center)

                # Update previous valid centers for future reference
                if valid_centers:
                    self.previous_valid_centers = valid_centers

            # Record data from this scan line
            if valid_centers:
                valid_lane_count += 1
                scan_line_counts.append(len(valid_centers))

                # Weight by vertical position (closer = more important)
                weight = 1.0 - (y_pos / h)

                for center in valid_centers:
                    all_lane_centers.append(center)
                    all_weights.append(weight)
            else:
                scan_line_counts.append(0)

        # Detect curves from scan line data
        is_curve = False
        if valid_lane_count >= 2:
            is_curve = self._detect_curve_from_scan_counts(scan_line_counts)
            self.is_curve_detected = is_curve

        # Update lane count estimate
        if all_lane_centers:
            # Get clustered centers for better lane count estimation
            clustered_centers = self._cluster_lane_centers(all_lane_centers)

            # In curves, be more conservative with lane count changes
            if is_curve:
                estimated_lane_count = self.last_lane_count
            else:
                estimated_lane_count = len(clustered_centers)

                # Apply stability factor for smooth transitions
                stability_factor = 0.9 if abs(estimated_lane_count - self.last_lane_count) > 0 else 0.8
                self.lane_count = int(stability_factor * self.last_lane_count +
                                      (1 - stability_factor) * estimated_lane_count)

            self.last_lane_count = self.lane_count

        # Return processed data and confidence
        return {
            'lane_centers': all_lane_centers,
            'weights': all_weights,
            'clustered_centers': self._cluster_lane_centers(all_lane_centers) if all_lane_centers else [],
            'lane_count': self.lane_count,
            'is_curve': is_curve,
            'curve_direction': self.curve_direction,
            'speed': 0  # Will be updated by caller if available
        }, confidence

    def _detect_curve_from_scan_counts(self, scan_counts):
        """Improved curve detection by analyzing differences in lane counts across scan lines"""
        if not scan_counts or len(scan_counts) < self.MIN_SCAN_LINES_FOR_CURVE:
            return False

        # Filter out zero counts (invalid scan lines)
        valid_counts = [count for count in scan_counts if count > 0]
        if len(valid_counts) < self.MIN_SCAN_LINES_FOR_CURVE:
            return False

        # Method 1: Check for significant differences in lane counts
        max_count = max(valid_counts)
        min_count = min(valid_counts)
        if max_count - min_count >= 1:
            return True

        # Method 2: Check first and last valid scan lines for differences
        if len(valid_counts) >= 2 and abs(valid_counts[0] - valid_counts[-1]) >= 1:
            return True

        # Method 3: Check for horizontal shifts in lane centers
        if hasattr(self, 'lane_history') and len(self.lane_history) >= 2:
            recent_centers = self.lane_history[-1]
            if recent_centers and len(recent_centers) > 1:
                # Calculate average position
                avg_pos = sum(recent_centers) / len(recent_centers)

                # Check previous frame's centers if available
                if len(self.lane_history) > 1 and self.lane_history[-2]:
                    prev_centers = self.lane_history[-2]
                    prev_avg = sum(prev_centers) / len(prev_centers)

                    # Large horizontal shift indicates curve
                    shift = avg_pos - prev_avg
                    if abs(shift) > self.CURVE_DETECTION_THRESHOLD * self.expected_lane_width:
                        # Set curve direction based on shift direction
                        self.curve_direction = 'right' if shift < 0 else 'left'
                        return True

        return False

    def _group_lane_pixels(self, lane_pixels, gap_threshold=20):
        """Group lane pixels into potential lane lines using efficient algorithm"""
        if not lane_pixels.size:
            return []

        # Find gaps larger than threshold
        gaps = np.diff(lane_pixels) >= gap_threshold
        # Get indices where groups start (including first pixel)
        group_starts = np.concatenate(([0], np.where(gaps)[0] + 1))
        # Get indices where groups end (including last pixel)
        group_ends = np.concatenate((np.where(gaps)[0], [len(lane_pixels) - 1]))

        # Create groups with at least 3 points
        groups = []
        for start, end in zip(group_starts, group_ends):
            if end - start + 1 >= 3:  # Minimum 3 points per group
                groups.append(lane_pixels[start:end + 1])

        return groups

    def _filter_groups_by_direction(self, groups, y_pos, lane_data):
        """Filter groups based on directional analysis for better curve handling"""
        # Skip analysis if near image edges
        if y_pos >= lane_data.shape[0] - 20 or y_pos <= 20:
            return groups

        h, w = lane_data.shape[:2] if len(lane_data.shape) > 2 else lane_data.shape
        filtered_groups = []

        for group in groups:
            # Calculate center of this group
            center_x = int(np.mean(group))

            # Check vertical continuity
            valid_points = 0
            for offset in range(-20, 21, 5):
                check_y = y_pos + offset

                # Skip if outside image bounds
                if check_y < 0 or check_y >= h:
                    continue

                # Search range depends on distance from current scan line
                search_range = 30 if abs(offset) > 10 else 15

                # Check for lane pixels in nearby rows
                check_row = lane_data[check_y, :]
                x_start = max(0, center_x - search_range)
                x_end = min(w, center_x + search_range + 1)

                # Use vectorized operation for efficiency
                if np.any(check_row[x_start:x_end] > 0):
                    valid_points += 1

            # Keep groups with good vertical continuity
            if valid_points >= 3:
                filtered_groups.append(group)

        # If filtering removed too many groups, revert to original
        if len(filtered_groups) < 2 and len(groups) >= 2:
            return groups

        return filtered_groups

    def _cluster_lane_centers(self, centers, threshold=30):
        """Cluster lane centers to identify unique lanes using a more efficient approach"""
        if not centers:
            return []

        # Sort centers
        centers = sorted(centers)

        # Use numpy for vectorized operations
        centers_array = np.array(centers)
        diff = np.diff(centers_array)

        # Identify break points where distance exceeds threshold
        break_points = np.where(diff >= threshold)[0]

        # Split into clusters
        if len(break_points) == 0:
            # All centers belong to one cluster
            return [np.mean(centers_array)]

        # Use break points to form clusters
        cluster_indices = np.split(np.arange(len(centers)), break_points + 1)

        # Calculate mean of each cluster
        return [np.mean(centers_array[indices]) for indices in cluster_indices]

    def _update_lane_history(self, processed_data, confidence):
        """Update lane history with current detection and adapt lane width estimate"""
        clustered_centers = processed_data.get('clustered_centers', [])

        if not clustered_centers:
            return

        # Update lane width estimate if we have multiple lanes
        if len(clustered_centers) >= 2:
            # Calculate lane widths
            sorted_centers = sorted(clustered_centers)
            widths = [sorted_centers[i + 1] - sorted_centers[i]
                      for i in range(len(sorted_centers) - 1)]

            # Filter for reasonable widths
            valid_widths = [w for w in widths if 60 < w < 200]

            if valid_widths:
                # Use median for robustness
                median_width = sorted(valid_widths)[len(valid_widths) // 2]

                # Adaptive learning rate based on confidence
                alpha = self.lane_width_alpha * confidence
                self.expected_lane_width = (1 - alpha) * self.expected_lane_width + alpha * median_width

        # Add to lane history with limited size
        self.lane_history.append(clustered_centers)
        if len(self.lane_history) > self.max_history:
            self.lane_history.pop(0)

        self.lane_confidence = confidence

    def _determine_target_lane_position(self, processed_data):
        """Determine target lane position based on detected lanes and road configuration"""
        clustered_centers = processed_data.get('clustered_centers', [])

        if not clustered_centers:
            # No centers detected, fall back to previous error
            return self.prev_error

        # Sort centers from left to right
        sorted_centers = sorted(clustered_centers)

        # Simplified lane selection logic
        lane_count = processed_data.get('lane_count', 2)
        target = None

        if lane_count >= 3:
            # On 3+ lane roads, target center lane
            if len(sorted_centers) >= 3:
                middle_index = len(sorted_centers) // 2
                target = sorted_centers[middle_index]
            else:
                # Estimate center position
                target = sum(sorted_centers) / len(sorted_centers)
        else:
            # On 2-lane roads, target right lane
            if len(sorted_centers) >= 2:
                target = sorted_centers[1]  # Second lane (right lane in 2-lane)
            else:
                target = sorted_centers[0]  # Only one lane detected

        # Calculate image center with camera offset
        image_width = 1000  # Default width
        if self.prev_lane_data is not None:
            _, image_width = self.prev_lane_data.shape[:2] if len(
                self.prev_lane_data.shape) > 2 else self.prev_lane_data.shape
        vehicle_center = image_width // 2 + self.camera_offset

        # Calculate error (difference from vehicle center)
        return target - vehicle_center

    def _use_historical_steering(self, game_steer=0.0):
        """Use historical steering data when no lanes detected, with current steering consideration"""
        # Apply more reduction when current steering is significant
        steer_reduction_factor = 0.7 - min(0.4, abs(game_steer) * 0.5)
        reduced_steering = self.prev_error * steer_reduction_factor

        if abs(reduced_steering) < self.lane_center_threshold:
            return None, 0
        elif reduced_steering > 0:
            return 'd', min(abs(reduced_steering) / 150, 0.05)
        else:
            return 'a', min(abs(reduced_steering) / 150, 0.05)

    def _apply_adaptive_pid(self, error, speed, game_steer=0.0):
        """Apply PID control with parameters that adapt to driving conditions
        and take into account current steering angle"""
        # Base PID parameters
        kp_base = 0.01
        ki_base = 0.0005
        kd_base = 0.008

        # Adapt parameters based on speed
        speed_factor = min(1.0, max(0.5, speed / 30.0)) if speed > 0 else 0.5
        kp = kp_base * (1 + 0.2 * (1 - speed_factor))  # More aggressive at lower speeds
        ki = ki_base * speed_factor  # Less integral at high speeds
        kd = kd_base * (1 + 0.5 * speed_factor)  # More dampening at higher speeds

        # Calculate steering rate limiting factor based on current angle
        steering_limit_factor = 1.0 - min(1.0, abs(game_steer) / 0.3) * 0.7

        # Smoother error with adaptive filtering
        alpha = 0.7 * steering_limit_factor
        smoothed_error = alpha * error + (1 - alpha) * self.prev_error

        # Anti-windup for integral term
        max_i_term = 10 * steering_limit_factor
        self.error_sum = max(min(self.error_sum + smoothed_error * steering_limit_factor, max_i_term), -max_i_term)

        # Calculate PID terms
        p_term = kp * smoothed_error
        i_term = ki * self.error_sum
        d_term = kd * (smoothed_error - self.prev_error)

        # Update previous error
        self.prev_error = smoothed_error

        # Calculate total steering
        steering = p_term + i_term + d_term

        # Dynamic steering time based on error magnitude
        error_magnitude = min(abs(smoothed_error) / 180, 0.07) * steering_limit_factor

        # Reduce magnitude when turning in same direction as current steering
        if (steering > 0 and game_steer > 0.1) or (steering < 0 and game_steer < -0.1):
            error_magnitude *= 0.7

        # Apply steering based on threshold
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
        # 4. 映射到 "left"/"right"
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
            cx = np.clip(cx, 0, W - 1)
            cy = np.clip(cy, 0, H - 1)

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
        game_steer = status.get("game_steer", 0)

        # Process vehicle detection data
        car_list = self._process_car_detection(status)

        # Process vehicle detection and lane categorization
        lane_status = status.get("lane_status") or {}
        cars_in_lanes = []
        if car_list and isinstance(lane_status, dict):
            categorized = self.categorize_cars_by_lane(lane_status, car_list)
            left_cars = [obj for obj, side in categorized if side == "left"]
            middle_cars = [obj for obj, side in categorized if side == "middle"]
            right_cars = [obj for obj, side in categorized if side == "right"]

            print("******************")
            for obj, side in categorized:
                log.debug(f"Vehicle: {obj['class_name']} in lane: {side}")

        # Check for traffic lights first (highest priority)
        if status.get("traffic_detect") and len(status["traffic_detect"]) > 0:
            traffic_cmd = self._process_traffic_light(status["traffic_detect"], speed, game_steer)
            if traffic_cmd and len(traffic_cmd) > 0:
                self.last_taken_frame = status.get("detect_frame", None)
                return traffic_cmd

        # Detect road conditions (curves, etc.)
        road_conditions = self._detect_road_conditions(lane_data, cars_in_lanes)

        # Calculate steering action - pass game_steer to consider current steering angle
        steering_action, steering_duration = self.calculate_steering(lane_data, game_steer)

        # Calculate acceleration based on speed and road conditions
        accel_action, accel_duration = self._calculate_acceleration(speed, road_conditions, game_steer)

        # Apply action smoothing
        action = self._create_combined_action(steering_action, steering_duration,
                                              accel_action, accel_duration, game_steer)

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
        """Detect road conditions like curves, obstacles based on lane data and cars"""
        conditions = {
            'is_curve': self.is_curve_detected,
            'curve_direction': self.curve_direction,
            'has_obstacle': False,
            'obstacle_side': None,
            'recommend_lane_change': False
        }

        # Additional curve detection from lane data shape
        if lane_data is not None and isinstance(lane_data, np.ndarray):
            h, w = lane_data.shape[:2] if len(lane_data.shape) > 2 else lane_data.shape

            # Sample points across multiple heights
            scan_positions = [int(h * 0.3), int(h * 0.5), int(h * 0.7)]
            scan_centers = []

            for y_pos in scan_positions:
                if y_pos >= h:
                    continue

                row = lane_data[y_pos, :]
                lane_pixels = np.where(row > 0)[0]

                if len(lane_pixels) > 5:
                    # Calculate center of lane pixels
                    center = np.mean(lane_pixels)
                    scan_centers.append((y_pos, center))

            # Detect curve from scan centers
            if len(scan_centers) >= 2:
                scan_centers.sort()

                # Calculate horizontal shift between scan lines
                shifts = []
                for i in range(1, len(scan_centers)):
                    prev_y, prev_x = scan_centers[i - 1]
                    curr_y, curr_x = scan_centers[i]

                    # Calculate normalized shift
                    vert_dist = curr_y - prev_y
                    horz_shift = curr_x - prev_x
                    normalized_shift = horz_shift / vert_dist if vert_dist > 0 else 0

                    shifts.append(normalized_shift)

                # Detect curve from average shift
                if shifts:
                    avg_shift = sum(shifts) / len(shifts)
                    if abs(avg_shift) > self.CURVE_DETECTION_THRESHOLD:
                        conditions['is_curve'] = True
                        conditions['curve_direction'] = 'right' if avg_shift < 0 else 'left'

                        # Update class variables for use elsewhere
                        self.is_curve_detected = True
                        self.curve_direction = conditions['curve_direction']

        # Detect obstacles from cars_in_lanes
        if cars_in_lanes:
            for car_obj, lane in cars_in_lanes:
                if not isinstance(car_obj, dict) or 'bbox' not in car_obj:
                    continue

                # Calculate obstacle proximity based on bounding box
                x1, y1, x2, y2 = car_obj['bbox']
                car_height = y2 - y1
                car_width = x2 - x1

                # Calculate relative position factors
                proximity_factor = car_height / 300  # Normalized by typical maximum height
                size_factor = car_width * car_height / (300 * 200)  # Normalized by typical car size

                # Combined threat assessment
                threat_level = proximity_factor * size_factor

                # If car is in our lane and close/large enough to be a threat
                if lane == "right" and self.lane_count <= 2 and threat_level > 0.15:
                    conditions['has_obstacle'] = True
                    conditions['obstacle_side'] = "front"

                    # Check for lane change recommendation based on curve state
                    if not conditions['is_curve']:
                        # Only recommend lane changes on straight segments
                        if not any(l == "left" for car, l in cars_in_lanes if isinstance(car, dict)):
                            conditions['recommend_lane_change'] = True

        return conditions

    def _calculate_acceleration(self, speed, road_conditions, game_steer=0.0):
        """Calculate acceleration action based on speed, road conditions and current steering"""
        # Base acceleration parameters
        if speed < 5:
            # Higher acceleration at low speeds
            accel_duration = self.base_accel_duration * 1.2
        else:
            # Reduce acceleration at higher speeds for stability
            accel_factor = max(0.6, min(1.0, 15.0 / speed)) if speed > 0 else 1.0
            accel_duration = self.base_accel_duration * accel_factor

        # Adjust for steering angle - reduce acceleration when turning significantly
        steer_magnitude = abs(game_steer)
        if steer_magnitude > 0.15:
            # Calculate turning factor - reduces acceleration as steering increases
            turn_factor = max(0.5, 1.0 - (steer_magnitude - 0.15) * 2.0)
            accel_duration *= turn_factor

            # For sharp turns, apply braking instead
            if steer_magnitude > 0.4 and speed > 40:
                return 's', 0.03 * min(1.0, steer_magnitude)  # Light braking proportional to turn sharpness

        # Adjust for road conditions
        if road_conditions.get('is_curve', False):
            # Reduce speed in curves
            curve_severity = steer_magnitude * 2.0 if steer_magnitude > 0.15 else 1.0

            if speed > 60:
                return 's', 0.03 * curve_severity  # Braking proportional to curve severity
            elif speed < 40:
                return 'w', accel_duration * 0.8 / curve_severity  # Gentler acceleration in curves
            else:
                return 'w', accel_duration * 0.5 / curve_severity  # Maintain speed carefully in curves

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

    def _create_combined_action(self, steering_action, steering_duration, accel_action, accel_duration, game_steer=0.0):
        """Create combined steering and acceleration action with improved smoothing"""
        # Only apply steering action if it's significant enough or in the opposite direction
        # of the current steering to correct course
        if steering_action:
            # If requesting same direction as current steering and magnitude is already high,
            # reduce the duration or cancel if not needed
            if (steering_action == 'd' and game_steer > 0.3) or (steering_action == 'a' and game_steer < -0.3):
                # Already turning strongly in requested direction, reduce or cancel
                steering_duration *= 0.5
                if steering_duration < 0.01:  # Below effectiveness threshold
                    steering_action = None

            # If correcting in opposite direction, maintain or slightly increase duration
            elif (steering_action == 'd' and game_steer < -0.1) or (steering_action == 'a' and game_steer > 0.1):
                # Correcting from opposite direction - maintain or increase slightly
                steering_duration *= 1.1

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

    def _process_traffic_light(self, traffic_light_data, speed=0, game_steer=0):
        """
        Process traffic light data and return appropriate action in the correct format.
        Enhanced braking for red lights to ensure vehicle stops completely.

        Args:
            traffic_light_data: List of detected traffic lights
            speed: Current vehicle speed
            game_steer: Current steering angle

        Returns:
            List of action commands in format [[direction:duration, movement:duration]] or [] if no action
        """
        # Initialize traffic light state if not exists
        if not hasattr(self, 'traffic_light_state'):
            self.traffic_light_state = {
                'active': False,
                'color': None,
                'start_time': time.time(),
                'last_detection_time': 0,
                'exit_timeout': 5.0,  # Seconds to maintain state after light disappears
                'stability_count': 0,
                'braking_level': 0,  # Track braking intensity level (0-5)
                'stop_detected': False  # Flag for complete stop detection
            }

        state = self.traffic_light_state
        current_time = time.time()

        # Detect most confident traffic light
        current_light = None
        highest_conf = 0.6  # Minimum confidence threshold

        for data in traffic_light_data:
            if isinstance(data, dict):
                confidence = data.get('confidence', 0)
                class_name = data.get('class_name', '').lower()
                print(class_name)

                if confidence > highest_conf:
                    if 'red' in class_name:
                        current_light = 'red'
                        highest_conf = confidence
                    elif 'green' in class_name:
                        current_light = 'green'
                        highest_conf = confidence
                    elif 'yellow' in class_name:
                        current_light = 'yellow'
                        highest_conf = confidence

        # Update detection time if light detected
        if current_light:
            state['last_detection_time'] = current_time

            # Reset braking level if color changes
            if state['color'] != current_light:
                if current_light == 'red':
                    # Initialize braking level based on current speed
                    state['braking_level'] = min(5, max(1, int(speed / 10)))
                    state['stop_detected'] = False
                elif current_light == 'green':
                    state['braking_level'] = 0
                    state['stop_detected'] = False

            state['color'] = current_light

            # Activate traffic light mode if not already active
            if not state['active']:
                state['active'] = True
                state['start_time'] = current_time
                state['stability_count'] = 0
                log.debug(f"Entering traffic light mode: {current_light}")

        # Check if we should exit traffic light mode
        if state['active'] and current_time - state['last_detection_time'] > state['exit_timeout']:
            if state['stability_count'] > 20:  # Exit only if stable for a while
                state['active'] = False
                log.debug("Exiting traffic light mode - timeout")
                return []

        # Return if no active traffic light mode
        if not state['active']:
            return []

        # Detect complete stop
        if speed < 0.5 and state['color'] == 'red':
            state['stop_detected'] = True
            log.debug("Vehicle has come to a complete stop at red light")

        # Handle based on light color
        if state['color'] == 'red':
            # Check if already stopped
            if state['stop_detected'] and speed < 1.0:
                # Maintain stop with light brake
                return [[f'none:0.03', f's:0.03']]

            # First correct steering if significant deviation
            if abs(game_steer) > 0.1 and not state['stop_detected']:
                # Correct steering before braking
                steer_key = 'a' if game_steer > 0 else 'd'
                steer_duration = min(0.05, abs(game_steer) * 0.1)
                # Add some braking while steering
                return [[f'{steer_key}:{steer_duration:.2f}', f's:{steer_duration:.2f}']]

            # Progressive braking based on speed
            if speed > 30:
                # High speed - strong braking
                brake_intensity = 0.15
                state['braking_level'] = 5
            elif speed > 20:
                # Medium speed - moderate braking
                brake_intensity = 0.12
                state['braking_level'] = 4
            elif speed > 10:
                # Low speed - lighter braking
                brake_intensity = 0.09
                state['braking_level'] = 3
            elif speed > 5:
                # Very low speed - gentle braking
                brake_intensity = 0.07
                state['braking_level'] = 2
            else:
                # Crawling - minimal braking to stop
                brake_intensity = 0.05
                state['braking_level'] = 1

            # Apply braking
            return [[f'none:{brake_intensity:.2f}', f's:{brake_intensity:.2f}']]

        elif state['color'] == 'yellow':
            # Yellow light behavior - similar to red but less aggressive

            # First correct steering if significant deviation
            if abs(game_steer) > 0.1:
                # Correct steering before braking
                steer_key = 'a' if game_steer > 0 else 'd'
                steer_duration = min(0.04, abs(game_steer) * 0.1)
                return [[f'{steer_key}:{steer_duration:.2f}', f's:{steer_duration:.2f}']]

            # Progressive braking based on speed (lighter than red)
            if speed > 30:
                brake_intensity = 0.10
            elif speed > 20:
                brake_intensity = 0.08
            elif speed > 10:
                brake_intensity = 0.06
            else:
                brake_intensity = 0.04

            # Apply braking
            return [[f'none:{brake_intensity:.2f}', f's:{brake_intensity:.2f}']]

        elif state['color'] == 'green':
            # Increment stability counter
            state['stability_count'] += 1

            # First correct steering if significant deviation
            if abs(game_steer) > 0.1 and state['stability_count'] < 10:
                steer_key = 'a' if game_steer > 0 else 'd'
                steer_duration = min(0.04, abs(game_steer) * 0.1)
                return [[f'{steer_key}:{steer_duration:.2f}', f'w:{steer_duration:.2f}']]

            # Handle speed based on current value
            if speed > 30:
                # Above speed limit - apply gentle brake
                return [[f'none:0.03', f's:0.03']]
            elif speed < 25:
                # Below target speed - accelerate gently
                accel_intensity = 0.05 * (1 - (speed / 25))  # Reduce acceleration as speed approaches 25
                accel_intensity = max(0.02, min(0.06, accel_intensity))  # Clamp between 0.02 and 0.06
                return [[f'none:{accel_intensity:.2f}', f'w:{accel_intensity:.2f}']]
            else:
                # Maintain speed within target range (25-30)
                return [[f'none:0.02', f'w:0.02']]

        # Default case - no action
        return []