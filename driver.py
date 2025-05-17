import numpy as np


class TruckController():
    def __init__(self):
        self.autodrive_active = False


    def get_drive_status(self):
        return self.autodrive_active

    def drive_mode_toggle(self):
        self.autodrive_active = not self.autodrive_active

    #根据车道和事件的综合因素，决定应该进行的操作
    def get_action(self, status):
        commands = []
        lane_data = status.get("lane_status")['lane_data']
        # traffic_light_data = status.get("traffic_classes")

        if lane_data is None:
            # If lane detection failed, stop the truck
            commands.append(['s', '0.05'])
            return commands

        # Analyze bottom third of the image for immediate lane position
        height, width = lane_data.shape
        bottom_section = lane_data[int(height * 2 / 3):, :]

        # Find lane markers (non-zero pixels)
        lane_pixels = np.where(bottom_section > 0)

        if len(lane_pixels[1]) == 0:
            # No lane detected, slow down
            commands.append(['s', '0.03'])
            return commands

        # traffic_status = self.process_traffic_light(traffic_light_data)
        # commands.append(traffic_status)

        # Calculate center of lane
        left_points = []
        right_points = []

        # Find left and right lane markers
        mid_point = width // 2
        for y, x in zip(lane_pixels[0], lane_pixels[1]):
            if x < mid_point:
                left_points.append(x)
            else:
                right_points.append(x)

        # Determine lane center
        if left_points and right_points:
            # Both lane markers visible
            left_edge = max(left_points)
            right_edge = min(right_points)
            lane_center = (left_edge + right_edge) // 2
        elif left_points:
            # Only left lane visible, estimate center
            left_edge = max(left_points)
            lane_center = left_edge + 50  # Estimated lane width
        elif right_points:
            # Only right lane visible, estimate center
            right_edge = min(right_points)
            lane_center = right_edge - 50  # Estimated lane width
        else:
            # Fallback
            lane_center = mid_point

        # Calculate deviation from center
        center_deviation = lane_center - mid_point

        # Determine steering command
        if abs(center_deviation) < 15:
            # Go straight - just accelerate
            commands.append(['w', '0.02'])
        elif center_deviation < 0:
            # Need to steer left
            steer_duration = min(abs(center_deviation) / 100, 0.5)  # Max 0.5 seconds
            commands.append(['a', f'{steer_duration:.1f}'])
            # Maintain forward motion
            commands.append(['w', '0.01'])
        else:
            # Need to steer right
            steer_duration = min(center_deviation / 100, 0.5)  # Max 0.5 seconds
            commands.append(['d', f'{steer_duration:.1f}'])
            # Maintain forward motion
            commands.append(['w', '0.01'])

        return commands


    def process_traffic_light(self, traffic_light_data):
        """处理YOLO检测到的红绿灯数据，返回是否需要停车"""
        # 如果data数组中有Red的字符串则返回ImmediateStop，有Green则返回启动，Yellow则减速
        for data in traffic_light_data:
            if "Red" in data:
                return ['s', '0.05']  # 停车0.5秒
            elif "Green" in data:
                return ['w', '0.02']
            elif "Yellow" in data:
                return ['s', '0.03']  # 减速0.3秒

        return None  # 默认不操作
